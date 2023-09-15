/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
REGISTER_OP("NvtxStart")
    .Input("inputs: T")
    .Input("null_input: float32")
    .Input("message: string")
    .Input("domain_name: string")
    .Output("output: T")
    .Output("marker_id: int64")
    .Output("domain_handle: int64")
    //.Attr("T: list({int32, int64, float32})")
    .Attr("T: list({int32, int64, float32}) >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for(int i = 0; i < c->num_inputs() - 3; ++i) {
        c->set_output(i, c->input(i));
        auto* handle_data = c->input_handle_shapes_and_types(i);
        if (handle_data != nullptr) {
          c->set_output_handle_shapes_and_types(i, *handle_data);
        }
      }
      c->set_output(c->num_inputs()-3, c->Scalar());
      c->set_output(c->num_inputs()-2, c->Scalar());
      return OkStatus();
    })
    .Doc(R"doc(
An identity graph node with a side effect of opening an NVTX marker.


Arguments
    inputs: A `Tensor` object that will be passed to `output`.
    null_input: A `float32 Tensor` object used as a trick to force gradient
                calculation. The tesnor is not used inside the op.
    message: A `String` message associated with this op.
    domain_name: A `String` domain name associated with this op.

Output
    output: The input `Tensor` passed to the output.
    marker_id: An NVTX marker id that is passed to `NvtxEnd`.
    domain_handle: An NVTX domain handler that is passed to `NvtxEnd`.
)doc");

REGISTER_OP("NvtxEnd")
    .Input("inputs: T")
    .Input("marker_id: int64")
    .Input("domain_handle: int64")
    .Input("grad_message: string")
    .Input("grad_domain_name: string")
    .Output("output: T")
    .Output("null_output: float32")
    .Attr("T: list({int32, int64, float32}) >= 0")
    //.Attr("T: list(type) >= 0 = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for(int i = 0; i < c->num_inputs() - 4; ++i) {
        c->set_output(i, c->input(i));
        auto* handle_data = c->input_handle_shapes_and_types(i);
        if (handle_data != nullptr) {
          c->set_output_handle_shapes_and_types(i, *handle_data);
        }
      }
      /*
      c->set_output(0, c->input(0));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      */
      c->set_output(c->num_inputs() - 4, c->Scalar());
      return OkStatus();
    })
    .Doc(R"doc(
An identity graph node with a side effect of closing an NVTX marker.


Arguments
    inputs: A `Tensor` object that will be passed to `output`.
    marker_id: An NVTX marker id that is recived from `NvtxStart`.
    domain_handle: An NVTX domain handler that is recived from `NvtxStart`.
    grad_message: A `String` message associated with this op gradient.
    grad_domain_name: A `String` domain name associated with this op gradient.

Output
    output: The input `Tensor` passed to the output.
    null_output: A `float32 Tensor` object used as a trick to force gradient
                 calculation. The tesnor is not used inside the op.
)doc");

