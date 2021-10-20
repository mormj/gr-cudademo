/* -*- c++ -*- */
/*
 * Copyright 2021 Josh.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDADEMO_MULTIPLY_CONST_H
#define INCLUDED_CUDADEMO_MULTIPLY_CONST_H

#include <gnuradio/sync_block.h>
#include <cudademo/api.h>

namespace gr {
namespace cudademo {

/*!
 * \brief <+description of block+>
 * \ingroup cudademo
 *
 */
class CUDADEMO_API multiply_const : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<multiply_const> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cudademo::multiply_const.
     *
     * To avoid accidental use of raw pointers, cudademo::multiply_const's
     * constructor is in a private implementation
     * class. cudademo::multiply_const::make is the public interface for
     * creating new instances.
     */
    static sptr make(float k);

    virtual void set_k(float k) = 0;
};

} // namespace cudademo
} // namespace gr

#endif /* INCLUDED_CUDADEMO_MULTIPLY_CONST_H */
