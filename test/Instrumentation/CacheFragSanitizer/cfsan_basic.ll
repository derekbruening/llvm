; Test basic CacheFragSanitizer instrumentation.
;
; RUN: opt < %s -cfsan -S | FileCheck %s

define i32 @loadWord(i32* %a) {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

; CHECK: @llvm.global_ctors = {{.*}}@cfsan.module_ctor

; CHECK:        %0 = ptrtoint i32* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 158329674399744
; CHECK-NEXT:   %3 = lshr i64 %2, 3
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i64 %0, 7
; CHECK-NEXT:   %7 = shl i64 15, %6
; CHECK-NEXT:   %8 = trunc i64 %7 to i8
; CHECK-NEXT:   %9 = and i8 %5, %8
; CHECK-NEXT:   %10 = icmp ne i8 %9, %8
; CHECK-NEXT:   br i1 %10, label %11, label %14
; CHECK:        %12 = or i8 %5, %8
; CHECK-NEXT:   %13 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %12, i8* %13
; CHECK-NEXT:   br label %14
; CHECK:        %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:   ret i32 %tmp1

; Ensure that cfsan converts intrinsics to calls:

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)

define void @memCpyTest(i8* nocapture %x, i8* nocapture %y) {
entry:
    tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %x, i8* %y, i64 16, i32 4, i1 false)
    ret void
; CHECK: define void @memCpyTest
; CHECK: call i8* @memcpy
; CHECK: ret void
}

define void @memMoveTest(i8* nocapture %x, i8* nocapture %y) {
entry:
    tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %x, i8* %y, i64 16, i32 4, i1 false)
    ret void
; CHECK: define void @memMoveTest
; CHECK: call i8* @memmove
; CHECK: ret void
}

define void @memSetTest(i8* nocapture %x) {
entry:
    tail call void @llvm.memset.p0i8.i64(i8* %x, i8 77, i64 16, i32 4, i1 false)
    ret void
; CHECK: define void @memSetTest
; CHECK: call i8* @memset
; CHECK: ret void
}

; CHECK: define internal void @cfsan.module_ctor()
; CHECK: call void @__cfsan_init()
