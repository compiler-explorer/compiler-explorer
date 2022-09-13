source_filename = "./example.ll"

define dso_local void @_Z4testi(i32 %arg) local_unnamed_addr {
bb:
  %i = icmp sgt i32 %arg, 0
  br i1 %i, label %.preheader, label %.loopexit

.loopexit: ; preds = %.preheader, %bb
  ret void

.preheader: ; preds = %bb, %.preheader
  %i1 = phi i32 [ %i2, %.preheader ], [ 0, %bb ]
  tail call void @_Z4calli(i32 %i1)
  %i2 = add nuw nsw i32 %i1, 1
  %i3 = icmp eq i32 %i2, %arg
  br i1 %i3, label %.loopexit, label %.preheader
}

declare dso_local void @_Z4calli(i32) local_unnamed_addr