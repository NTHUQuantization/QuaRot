; ModuleID = '/tmp/verification_rsqrt_overload_probe.hip'
source_filename = "/tmp/verification_rsqrt_overload_probe.hip"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_22b58b6fa8bce316 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_22b58b6fa8bce316 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define protected amdgpu_kernel void @rsqrt_overload_probe(ptr addrspace(1) noundef readonly captures(none) %0, ptr addrspace(1) noundef writeonly captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %4 = zext nneg i32 %3 to i64
  %5 = getelementptr inbounds nuw float, ptr addrspace(1) %0, i64 %4
  %6 = load float, ptr addrspace(1) %5, align 4, !tbaa !6
  %7 = fpext contract float %6 to double
  %8 = tail call double @llvm.amdgcn.rsq.f64(double %7)
  %9 = fneg double %7
  %10 = fmul double %8, %9
  %11 = tail call double @llvm.fma.f64(double %10, double %8, double 1.000000e+00)
  %12 = fmul double %8, %11
  %13 = tail call double @llvm.fma.f64(double %11, double 3.750000e-01, double 5.000000e-01)
  %14 = tail call double @llvm.fma.f64(double %12, double %13, double %8)
  %15 = tail call i1 @llvm.is.fpclass.f64(double %8, i32 384)
  %16 = select i1 %15, double %14, double %8
  %17 = fptrunc contract double %16 to float
  %18 = shl nuw nsw i32 %3, 1
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %19
  store float %17, ptr addrspace(1) %20, align 4, !tbaa !6
  %21 = fcmp olt float %6, 0x3810000000000000
  %22 = fmul float %6, 0x4170000000000000
  %23 = select i1 %21, float %22, float %6
  %24 = tail call float @llvm.amdgcn.rsq.f32(float %23)
  %25 = fmul float %24, 4.096000e+03
  %26 = select i1 %21, float %25, float %24
  %27 = getelementptr inbounds nuw i8, ptr addrspace(1) %20, i64 4
  store float %26, ptr addrspace(1) %27, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.amdgcn.rsq.f64(double) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f64(double, i32 immarg) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.rsq.f32(float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!opencl.ocl.version = !{!5}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{!"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"}
!5 = !{i32 2, i32 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
