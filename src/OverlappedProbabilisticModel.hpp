/*
 *       OverlappedProbabilisticModel.hpp
  *
 *       Copyright 2011 Waldir Caro <waldirc@ime.usp.br>
 *                      Alan Mitchell Durham <aland@usp.br>
 *
 *       This program is free software; you can redistribute it and/or modify
 *       it under the terms of the GNU  General Public License as published by
 *       the Free Software Foundation; either version 3 of the License, or
 *       (at your option) any later version.
 *
 *       This program is distributed in the hope that it will be useful,
 *       but WITHOUT ANY WARRANTY; without even the implied warranty of
 *       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *       GNU General Public License for more details.
 *
 *       You should have received a copy of the GNU General Public License
 *       along with this program; if not, write to the Free Software
 *       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *       MA 02110-1301, USA.
 */


#ifndef OVERLAP_MODEL_HPP
#define OVERLAP_MODEL_HPP

#include "crossplatform.hpp"

#include "ProbabilisticModelDecorator.hpp"
#include "NeuralNetworkModel.hpp"
#include "Symbol.hpp"
namespace tops {

  class DLLEXPORT OverlappedProbabilisticModel : public ProbabilisticModelDecorator {
  private:
    
    int _seqLength;
    int _left_overlap;
    int _right_overlap;
    int _model_virtual_size;
    
    AlphabetPtr revAlphabet;
    DoubleVector _overlapped_sum_array;
  public:
    OverlappedProbabilisticModel(ProbabilisticModelPtr m) : ProbabilisticModelDecorator(m)
    {      
    }
    /*OverlappedProbabilisticModel() : ProbabilisticModelDecorator()
    {      
    }*/ 
      
    virtual ~OverlappedProbabilisticModel(){};
    virtual double evaluate(const Sequence & s, unsigned int begin, unsigned int end) const;
    virtual Sequence & choose(Sequence & h, int size) const ;
    virtual Sequence & choose(Sequence &h, int initial_phase, int size) const;
    virtual double prefix_sum_array_compute(int begin , int end) ;
    virtual double prefix_sum_array_compute(int begin , int end, int phase);
    virtual bool initialize_prefix_sum_array(const Sequence & s, int phase);
    virtual bool initialize_prefix_sum_array(const Sequence & s) ;
    virtual std::string str() const;
    std::string model_name () const;
    virtual ProbabilisticModelParameters parameters() const ;
    virtual void initialize(int left_overlap, int right_overlap, int model_virtual_size) ;

  };
  typedef boost::shared_ptr<OverlappedProbabilisticModel> OverlappedProbabilisticModelPtr;
}

#endif