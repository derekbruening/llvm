//===-- CacheFragSanitizer.cpp - performance tuner ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of CacheFragSanitizer, a performance tuning tool
// that detects cache fragmentation.
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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cfsan"

static cl::opt<bool>  ClInstrumentMemoryAccesses(
    "cfsan-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>  ClInstrumentMemIntrinsics(
    "cfsan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool>  ClBlindShadowWrites(
    "cfsan-blind-shadow-writes", cl::init(false),
    cl::desc("Write without checking the current shadow value)"), cl::Hidden);

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumFastpaths, "Number of instrumented fastpaths");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");

static const char *const kCfsanModuleCtorName = "cfsan.module_ctor";
static const char *const kCfsanInitName = "__cfsan_init";

// We must keep these consistent with the cfsan runtime:
static const uint64_t kShadowScale = 3; // 8B:1B or 1B:1b
static const uint64_t kShadowMask = 0x00000fffffffffffull;
static const uint64_t kShadowOffs = 0x0000120000000000ull;
static const uint64_t kShadowModeBits = 2; // Bottom 2 bits

namespace {

/// CacheFragSanitizer: instrument each module to find cache fragmentation.
class CacheFragSanitizer : public FunctionPass {
 public:
  CacheFragSanitizer() : FunctionPass(ID) {}
  const char *getPassName() const override;
  bool runOnFunction(Function &F) override;
  bool doInitialization(Module &M) override;
  static char ID;

 private:
  void initializeCallbacks(Module &M);
  bool instrumentLoadOrStore(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  bool ignoreMemoryAccess(Instruction *I);
  int getMemoryAccessFuncIndex(Value *Addr, const DataLayout &DL);
  Value *appToShadow(Value *Shadow, IRBuilder<> &IRB);
  bool instrumentFastpath(Instruction *I, const DataLayout &DL, bool IsWrite,
                          Value *Addr, unsigned Alignment);

  LLVMContext *Cxt;
  Type *IntptrTy;
  IntegerType *OrdTy;
  // Our slowpath involves callouts to the runtime library.
  // Access sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  Function *CfsanRead[kNumberOfAccessSizes];
  Function *CfsanWrite[kNumberOfAccessSizes];
  Function *CfsanUnalignedRead[kNumberOfAccessSizes];
  Function *CfsanUnalignedWrite[kNumberOfAccessSizes];
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *CfsanCtorFunction;
};
}  // namespace

char CacheFragSanitizer::ID = 0;
INITIALIZE_PASS(CacheFragSanitizer, "cfsan",
    "CacheFragSanitizer: finds cache fragmentation.",
    false, false)

const char *CacheFragSanitizer::getPassName() const {
  return "CacheFragSanitizer";
}

FunctionPass *llvm::createCacheFragSanitizerPass() {
  return new CacheFragSanitizer();
}

void CacheFragSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  // Initialize the callbacks.
  OrdTy = IRB.getInt32Ty();
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    const unsigned ByteSize = 1U << i;
    std::string ByteSizeStr = utostr(ByteSize);
    // FIXME: inline the most common (i.e., aligned and frequent sizes)
    // read + write instrumentation instead of using callouts.
    SmallString<32> ReadName("__cfsan_read" + ByteSizeStr);
    CfsanRead[i] = checkSanitizerInterfaceFunction(M.getOrInsertFunction(
        ReadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));

    SmallString<32> WriteName("__cfsan_write" + ByteSizeStr);
    CfsanWrite[i] = checkSanitizerInterfaceFunction(M.getOrInsertFunction(
        WriteName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));

    SmallString<64> UnalignedReadName("__cfsan_unaligned_read" + ByteSizeStr);
    CfsanUnalignedRead[i] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            UnalignedReadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));

    SmallString<64> UnalignedWriteName("__cfsan_unaligned_write" + ByteSizeStr);
    CfsanUnalignedWrite[i] =
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

bool CacheFragSanitizer::doInitialization(Module &M) {
  Cxt = &(M.getContext());
  const DataLayout &DL = M.getDataLayout();
  IntptrTy = DL.getIntPtrType(M.getContext());
  std::tie(CfsanCtorFunction, std::ignore) = createSanitizerCtorAndInitFunctions(
      M, kCfsanModuleCtorName, kCfsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{});

  appendToGlobalCtors(M, CfsanCtorFunction, 0);

  return true;
}

bool CacheFragSanitizer::ignoreMemoryAccess(Instruction *I) {
  // We'd like to know about cache fragmentation in vtable accesses and
  // constant data references, so we do not currently ignore anything.
  return false;
}

bool CacheFragSanitizer::runOnFunction(Function &F) {
  // This is required to prevent instrumenting the call to __cfsan_init from
  // within the module constructor.
  if (&F == CfsanCtorFunction)
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

  if (ClInstrumentMemoryAccesses) {
    for (auto Inst : LoadsAndStores) {
      Res |= instrumentLoadOrStore(Inst, DL);
    }
  }

  if (ClInstrumentMemIntrinsics) {
    for (auto Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(Inst);
    }
  }

  return Res;
}

bool CacheFragSanitizer::instrumentLoadOrStore(Instruction *I,
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
  if (IsWrite)
    NumInstrumentedWrites++;
  else
    NumInstrumentedReads++;
  if (instrumentFastpath(I, DL, IsWrite, Addr, Alignment)) {
    NumFastpaths++;
    return true;
  }
  if (Alignment == 0 || Alignment >= 8 || (Alignment % (TypeSize / 8)) == 0)
    OnAccessFunc = IsWrite ? CfsanWrite[Idx] : CfsanRead[Idx];
  else
    OnAccessFunc = IsWrite ? CfsanUnalignedWrite[Idx] : CfsanUnalignedRead[Idx];
  IRB.CreateCall(OnAccessFunc, IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()));
  return true;
}

// It's simplest to replace the memset/memmove/memcpy intrinsics with
// calls that the runtime library intercepts.
// Our pass is late enough that calls should not turn back into intrinsics.
bool CacheFragSanitizer::instrumentMemIntrinsic(Instruction *I) {
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

int CacheFragSanitizer::getMemoryAccessFuncIndex(Value *Addr,
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

Value *CacheFragSanitizer::appToShadow(Value *Shadow, IRBuilder<> &IRB) {
  // Shadow = ((App & Mask) + Offs) >> Scale
  Shadow = IRB.CreateAnd(Shadow, ConstantInt::get(IntptrTy, kShadowMask));
  uint64_t Offs = kShadowOffs << kShadowScale;
  Shadow = IRB.CreateAdd(Shadow, ConstantInt::get(IntptrTy, Offs));
  if (kShadowScale > 0)
    Shadow = IRB.CreateLShr(Shadow, kShadowScale);
  return Shadow;
}

bool CacheFragSanitizer::instrumentFastpath(Instruction *I,
                                            const DataLayout &DL,
                                            bool IsWrite,
                                            Value *Addr,
                                            unsigned Alignment) {
  assert(kShadowScale == 3); // The code below assumes this
  // TODO(bruening): as can be seen below, the 1B:1b shadow mapping
  // requires complex instrumentation.  We may want to consider a simpler
  // 4B:1B mapping and give up byte granularity.
  IRBuilder<> IRB(I);
  Type *OrigTy = cast<PointerType>(Addr->getType())->getElementType();
  const uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (!(TypeSize == 8 ||
        // Ensure the shadow is limited to a single byte
        (TypeSize == 16 && Alignment == 2) ||
        (TypeSize == 32 && Alignment == 4) ||
        (TypeSize == 64 && Alignment == 8)))
    return false;

  // We inline instrumentation to set the corresponding shadow bit for each
  // byte touched by the application.

  Value *AddrPtr = IRB.CreatePointerCast(Addr, IntptrTy);
  Value *ShadowPtr = appToShadow(AddrPtr, IRB);
  Type *ShadowTy = IntegerType::get(*Cxt, 8U);
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *ShadowVal;
  
  if (TypeSize == 64) {
    // No bit manipulation necessary
    if (!ClBlindShadowWrites) {
      ShadowVal = IRB.CreateLoad(IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy));
      Value *Cmp = IRB.CreateICmpNE(ConstantInt::get(ShadowTy, -1),
                                    ShadowVal);
      TerminatorInst *CmpTerm = SplitBlockAndInsertIfThen(Cmp, I, false);
      // FIXME: do I need to call SetCurrentDebugLocation?
      IRB.SetInsertPoint(CmpTerm);
    }
    IRB.CreateStore(ConstantInt::get(ShadowTy, -1),
                    IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy));
    if (!ClBlindShadowWrites)
      IRB.SetInsertPoint(I);
    return true;
  }

  ShadowVal = IRB.CreateLoad(IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy));

  // Mask = ((1U << (TypeSize/8)) -1) << (AddrPtr & 0x7)
  Value *Mask =
    IRB.CreateAnd(AddrPtr, ConstantInt::get(AddrPtr->getType(), 0x7));
  Mask = IRB.CreateShl(ConstantInt::get(AddrPtr->getType(),
                                        (1U << (TypeSize/8)) -1),
                       Mask);
  Mask = IRB.CreateIntCast(Mask, ShadowTy, false);

  if (!ClBlindShadowWrites) {
    Value *Bits = IRB.CreateAnd(ShadowVal, Mask);
    Value *Cmp = IRB.CreateICmpNE(Bits, Mask);
    TerminatorInst *CmpTerm = SplitBlockAndInsertIfThen(Cmp, I, false);
    // FIXME: do I need to call SetCurrentDebugLocation?
    IRB.SetInsertPoint(CmpTerm);
  }

  ShadowVal = IRB.CreateOr(ShadowVal, Mask);
  IRB.CreateStore(ShadowVal,
                  IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy));
  
  if (!ClBlindShadowWrites)
    IRB.SetInsertPoint(I);
  return true;
}
