kind: Fixed: norm2 backward gives 0 for an all-zero group (it gave NaN)
---
The norm2 (L2) reduction backward computed `dy * x / y`; when every element of a reduced group is zero, `y == 0` and the result was 0/0 = NaN. torch masks that case to 0 (the subgradient at the origin). It now returns 0 there; a NaN norm still gives NaN. f32, f16, bf16, f64. Device test `reduce_norm2_backward_zero`: red on the old kernel, green on the fix.
