//===-- DeadStoreTuner.cpp - performance tuner ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DeadStoreTuner, a performance tuning tool
// that detects dead writes.
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//      - Later we may inline some of the instrumentation.
//   - Turn mem{set,cpy,move} instrinsics into library calls.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "dstune"

static cl::opt<bool>  ClInstrumentMemoryAccesses(
    "dstune-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>  ClInstrumentMemIntrinsics(
    "dstune-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumOmittedReadsFromConstantGlobals,
          "Number of reads from constant globals");
STATISTIC(NumOmittedVtableAccesses, "Number of vtable reads");

static const char *const kDstuneModuleCtorName = "dstune.module_ctor";
static const char *const kDstuneInitName = "__dstune_init";

namespace {

/// DeadStoreTuner: instrument the code in module to find dead stores.
class DeadStoreTuner : public FunctionPass {
 public:
  DeadStoreTuner() : FunctionPass(ID) {}
  const char *getPassName() const override;
  bool runOnFunction(Function &F) override;
  bool doInitialization(Module &M) override;
  static char ID;

 private:
  void initializeCallbacks(Module &M);
  bool instrumentLoadOrStore(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  bool ignoreMemoryAccess(Instruction *I);
  bool addrPointsToConstantData(Value *Addr);
  int getMemoryAccessFuncIndex(Value *Addr, const DataLayout &DL);

  Type *IntptrTy;
  IntegerType *OrdTy;
  // Our initial instrumentation inserts callouts to the runtime library.
  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  Function *DstuneRead[kNumberOfAccessSizes];
  Function *DstuneWrite[kNumberOfAccessSizes];
  Function *DstuneUnalignedRead[kNumberOfAccessSizes];
  Function *DstuneUnalignedWrite[kNumberOfAccessSizes];
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *DstuneCtorFunction;
};
}  // namespace

char DeadStoreTuner::ID = 0;
INITIALIZE_PASS(DeadStoreTuner, "dstune",
    "DeadStoreTuner: finds dead stores.",
    false, false)

const char *DeadStoreTuner::getPassName() const {
  return "DeadStoreTuner";
}

FunctionPass *llvm::createDeadStoreTunerPass() {
  return new DeadStoreTuner();
}

void DeadStoreTuner::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  // Initialize the callbacks.
  OrdTy = IRB.getInt32Ty();
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    const unsigned ByteSize = 1U << i;
    std::string ByteSizeStr = utostr(ByteSize);
    // FIXME: inline the most common (i.e., aligned and frequent sizes)
    // read + write instrumentation instead of using callouts.
    SmallString<32> ReadName("__dstune_read" + ByteSizeStr);
    DstuneRead[i] = checkSanitizerInterfaceFunction(M.getOrInsertFunction(
        ReadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));

    SmallString<32> WriteName("__dstune_write" + ByteSizeStr);
    DstuneWrite[i] = checkSanitizerInterfaceFunction(M.getOrInsertFunction(
        WriteName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));

    SmallString<64> UnalignedReadName("__dstune_unaligned_read" + ByteSizeStr);
    DstuneUnalignedRead[i] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            UnalignedReadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));

    SmallString<64> UnalignedWriteName("__dstune_unaligned_write" + ByteSizeStr);
    DstuneUnalignedWrite[i] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            UnalignedWriteName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
  }
  MemmoveFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemcpyFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemsetFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt32Ty(), IntptrTy, nullptr));
}

bool DeadStoreTuner::doInitialization(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  IntptrTy = DL.getIntPtrType(M.getContext());
  std::tie(DstuneCtorFunction, std::ignore) = createSanitizerCtorAndInitFunctions(
      M, kDstuneModuleCtorName, kDstuneInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{});

  appendToGlobalCtors(M, DstuneCtorFunction, 0);

  return true;
}

static bool isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

bool DeadStoreTuner::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->isConstant()) {
      // Reads from constant globals cannot affect dead stores.
      NumOmittedReadsFromConstantGlobals++;
      return true;
    }
  }
  return false;
}

bool DeadStoreTuner::ignoreMemoryAccess(Instruction *I) {
  if (LoadInst *Load = dyn_cast<LoadInst>(I)) {
    Value *Addr = Load->getPointerOperand();
    if (addrPointsToConstantData(Addr)) {
      // Addr points to some constant data -- it cannot affect dead stores.
      return true;
    }
    if (isVtableAccess(Load)) {
      // We likely cannot eliminate vtable ptr writes as that would violate
      // the standard, so we skip analyzing them.
      NumOmittedVtableAccesses++;
      return true;
    }
  }
  return false;
}

bool DeadStoreTuner::runOnFunction(Function &F) {
  // This is required to prevent instrumenting the call to __dstune_init from
  // within the module constructor.
  if (&F == DstuneCtorFunction)
    return false;
  initializeCallbacks(*F.getParent());
  SmallVector<Instruction*, 8> LoadsAndStores;
  SmallVector<Instruction*, 8> MemIntrinCalls;
  bool Res = false;
  const DataLayout &DL = F.getParent()->getDataLayout();

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if ((isa<LoadInst>(Inst) || isa<StoreInst>(Inst) ||
           isa<AtomicRMWInst>(Inst) || isa<AtomicCmpXchgInst>(Inst)) &&
          !ignoreMemoryAccess(&Inst))
        LoadsAndStores.push_back(&Inst);
      else if (isa<MemIntrinsic>(Inst))
        MemIntrinCalls.push_back(&Inst);
    }
  }

  if (ClInstrumentMemoryAccesses)
    for (auto Inst : LoadsAndStores) {
      Res |= instrumentLoadOrStore(Inst, DL);
    }

  if (ClInstrumentMemIntrinsics)
    for (auto Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(Inst);
    }

  // XXX: can we skip accesses to promotable allocas?

  return Res;
}

bool DeadStoreTuner::instrumentLoadOrStore(Instruction *I,
                                           const DataLayout &DL) {
  IRBuilder<> IRB(I);
  bool IsWrite;
  Value *Addr;
  unsigned Alignment;
  if (LoadInst *Load = dyn_cast<LoadInst>(I)) {
    IsWrite = false;
    Alignment = Load->getAlignment();
    Addr = Load->getPointerOperand();
  } else if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
    IsWrite = true;
    Alignment = Store->getAlignment();
    Addr = Store->getPointerOperand();
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    IsWrite = true;
    Alignment = 0;
    Addr = RMW->getPointerOperand();
  } else if (AtomicCmpXchgInst *Xchg = dyn_cast<AtomicCmpXchgInst>(I)) {
    IsWrite = true;
    Alignment = 0;
    Addr = Xchg->getPointerOperand();
  } else
    llvm_unreachable("Unsupported mem access type");

  int Idx = getMemoryAccessFuncIndex(Addr, DL);
  if (Idx < 0)
    return false;
  Type *OrigTy = cast<PointerType>(Addr->getType())->getElementType();
  const uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  Value *OnAccessFunc = nullptr;
  if (Alignment == 0 || Alignment >= 8 || (Alignment % (TypeSize / 8)) == 0)
    OnAccessFunc = IsWrite ? DstuneWrite[Idx] : DstuneRead[Idx];
  else
    OnAccessFunc = IsWrite ? DstuneUnalignedWrite[Idx] : DstuneUnalignedRead[Idx];
  IRB.CreateCall(OnAccessFunc, IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()));
  if (IsWrite) NumInstrumentedWrites++;
  else         NumInstrumentedReads++;
  return true;
}

// It's simplest to replace the memset/memmove/memcpy intrinsics with
// calls that the runtime library intercepts.
// Our pass is late, so the calls should not turn back into intrinsics.
bool DeadStoreTuner::instrumentMemIntrinsic(Instruction *I) {
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    IRB.CreateCall(
        MemsetFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    IRB.CreateCall(
        isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(M->getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  }
  return false;
}

int DeadStoreTuner::getMemoryAccessFuncIndex(Value *Addr,
                                              const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 &&
      TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    NumAccessesWithBadSize++;
    // Ignore all unusual sizes.
    return -1;
  }
  size_t Idx = countTrailingZeros(TypeSize / 8);
  assert(Idx < kNumberOfAccessSizes);
  return Idx;
}
