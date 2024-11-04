/*
 *       OverlappedProbabilisticModel.cpp
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


#include "ProbabilisticModelCreatorClient.hpp"
#include "ConfigurationReader.hpp"
#include "OverlappedProbabilisticModel.hpp"
#include "NeuralNetworkModel.hpp"
#include "Alphabet.hpp"
#include "Symbol.hpp"
namespace tops {


  double OverlappedProbabilisticModel::evaluate(const Sequence & s, unsigned int begin, unsigned int end) const{
    Sequence revCompSeq;
    
    double result = subModel()->evaluate(revCompSeq, begin - _left_overlap, end + _right_overlap);
    return result;
  }


  Sequence & OverlappedProbabilisticModel::choose(Sequence & h, int size) const {
    //not implemented. Cannot do overlaps with choose, as a region would have to be "cosen"
    //by more than one model
    std::cerr << "ERROR could not use choose with overlapped models\n";
    exit(-1);
    return h;
  }


  Sequence & OverlappedProbabilisticModel::choose(Sequence &h, int initial_phase, int size) const{
    //not implemented. Cannot do overlaps with choose, as a region would have to be "cosen"
    //by more than one model
    std::cerr << "ERROR could not use choose with overlapped models\n";
    exit(-1);
    return h; 
  }


  double OverlappedProbabilisticModel::prefix_sum_array_compute(int begin , int end){
    int length = end - begin + 1;
    double r;
    if (length != _model_virtual_size){
        r = -HUGE ;//check the log library calls
    }
    r = _overlapped_sum_array[begin];
    return r;
  }
  double OverlappedProbabilisticModel::prefix_sum_array_compute(int begin , int end, int phase){
    return prefix_sum_array_compute(begin, end);
  }
 
  bool OverlappedProbabilisticModel::initialize_prefix_sum_array(const Sequence & s, int phase){    
    return initialize_prefix_sum_array(s);
  }

  bool OverlappedProbabilisticModel::initialize_prefix_sum_array(const Sequence & s) {
    bool submodel_initialize = subModel()->initialize_prefix_sum_array(s);
    
    DoubleVector submodel_sum_array = (boost::dynamic_pointer_cast<NeuralNetworkModel>(subModel()))->getScores();
    _overlapped_sum_array.resize(_left_overlap + submodel_sum_array.size() + _right_overlap + _model_virtual_size - 1);
    //shift_right_fill_log_zero(_left_overlap, _my_summ_array)
    //  overlap to the left
    size_t i;
    for (i = 0; i < _left_overlap; i++)
      _overlapped_sum_array[i] = -HUGE;
    //  insert submodel scores
    //_overlapped_sum_array.insert(_overlapped_sum_array.begin() + i - 1, submodel_sum_array.begin(), submodel_sum_array.end());
    for (size_t j = 0; j < submodel_sum_array.size(); j++)
      _overlapped_sum_array[i++] = submodel_sum_array[j];
    
    //  overlap to the right
    for (i += submodel_sum_array.size(); i < _overlapped_sum_array.size(); i++)
      _overlapped_sum_array[i] = -HUGE;

    //std::cerr << "[INFO] submodel = " << submodel_sum_array.size() << "\n";
    //std::cerr << "[INFO] decorator = " << _overlapped_sum_array.size() << "\n";
    return submodel_initialize;
 }
  std::string OverlappedProbabilisticModel::model_name () const {
    return "OverlappedModel";
  }

  std::string OverlappedProbabilisticModel::str() const{

    std::stringstream out;
    out << "model_name = \"" << model_name() << "\"" << std::endl;
    std::string modelname = ProbabilisticModelDecorator::subModelName();

    out << "left_overlap = " << _left_overlap << std::endl;
    out << "rigth_overlap = " << _right_overlap << std::endl;
    out << "signal_size = " << _model_virtual_size << std::endl;

    if(modelname.length() > 0) // the submodel was specified in an external file
      out << "model = \"" << modelname << "\"" << std::endl ;
    else // the submodel was specified within the decorator
      out << "model = [" << subModel()->str() << "]" << std::endl ;
    return out.str();
  }


  ProbabilisticModelParameters OverlappedProbabilisticModel::parameters() const
  {
    ProbabilisticModelParameters p ;
    p.add("model_name", StringParameterValuePtr(new StringParameterValue(model_name())));
    std::string modelname = ProbabilisticModelDecorator::subModelName();
    if(modelname.length() > 0)
      p.add("model", StringParameterValuePtr( new StringParameterValue(modelname)));
    else
      p.add("model", ProbabilisticModelParameterListValuePtr (new ProbabilisticModelParameterListValue(subModel()->parameters())));
    return p;
  }

  void OverlappedProbabilisticModel::initialize(int left_overlap, int right_overlap, int model_virtual_size)
  {
    _left_overlap = left_overlap;
    _right_overlap = right_overlap;
    _model_virtual_size = model_virtual_size;

    _overlapped_sum_array = {}; // empty scores
    
    //std::cerr << "$" << subModel()->str() << "$";
    //std::cerr << this->str() << "$";
  }

}