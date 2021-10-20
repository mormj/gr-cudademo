/* -*- c++ -*- */
/*
 * Copyright 2021 Josh.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDADEMO_MULTIPLY_CONST_IMPL_H
#define INCLUDED_CUDADEMO_MULTIPLY_CONST_IMPL_H

#include <cudademo/multiply_const.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {
namespace cudademo {

class multiply_const_impl : public multiply_const
{
private:
    float d_k;

    cudaStream_t d_stream;
    int d_min_grid_size;
    int d_block_size;

    float *d_dev_in;
    float *d_dev_out;

    size_t d_max_buffer_size = 65536;

public:
    multiply_const_impl(float k);
    ~multiply_const_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);

    virtual void set_k(float k)
    {
        d_k = k;
    }
};

} // namespace cudademo
} // namespace gr

#endif /* INCLUDED_CUDADEMO_MULTIPLY_CONST_IMPL_H */
