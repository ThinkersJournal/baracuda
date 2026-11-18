kind: fix graph node-list truncation and cuVS null params
---
`Graph::edges`, `GraphNode::dependencies` and `GraphNode::dependent_nodes` now truncate the buffer to the count the driver returned on the second call, so a graph that shrank between the size query and the fetch no longer yields null `GraphNode`s. `IvfFlat` build and search now reject a null params pointer from `*_params_create` instead of dereferencing it. Each fix has a unit test that fails with the fix removed (verified by sabotage). Resolves CodeQL alerts #1, #2, #4, #5, #7, #8.
