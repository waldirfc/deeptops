/*
 *       TrainHMMBaumWelch.hpp
 *
 *       Copyright 2011 Andre Yoshiaki Kashiwabara <akashiwabara@usp.br>
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

#ifndef TRAIN_BAUM_WELCH_HPP
#define TRAIN_BAUM_WELCH_HPP

#include "ProbabilisticModel.hpp"
#include "ProbabilisticModelCreator.hpp"
#include "ConfigurationReader.hpp"


namespace tops {

  //! Creates a HMM using Baum-Welch
  class TrainHMMBaumWelch : public ProbabilisticModelCreator {
  public:
    TrainHMMBaumWelch () {}
    virtual ~TrainHMMBaumWelch () {};
    //! Creates a probability model 
    /*! \param parameters is a set of parameters that is utilized to build the model */
    virtual ProbabilisticModelPtr create( ProbabilisticModelParameters & parameters) const ;

    //! Provides a help
    virtual std::string help() const {
      std::string s;
      return s;
    }


  };
  typedef boost::shared_ptr <TrainHMMBaumWelch> TrainHMMBaumWelchPtr ;
};


#endif
