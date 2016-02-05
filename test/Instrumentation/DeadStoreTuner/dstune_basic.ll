; FIXME: create a real test
; Test basic DeadStoreTuner instrumentation.
;
; RUN: opt < %s -dstune -S | FileCheck %s
