#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "json.hpp"
#include <iostream>

namespace ml
{
  /**
   * Reads input parameters from a file.
   * Only json file format is currently supported for 
   * the file containing user-specified options.
   *
   * This class also reads data from a specified file.
   * The first line of the data file must contain 
   * two integers specifying the number of columns 
   * containing outputs. The remaining columns are assumed to 
   * contain data for features.
   */
  class ReadInput
  {
  public:

    ReadInput() { };
    ~ReadInput() { };

    /**
     *Read input parameters from a file in JSOM format.
     */
    nlohmann::json_abi_v3_11_3::json readJson(std::string inputFileName)
    {

      std::ifstream inp;
      inp.open(inputFileName);
      std::stringstream buffer;
      buffer << inp.rdbuf();
      auto inputParameters = nlohmann::json::parse(buffer.str());
      inp.close();
      return inputParameters;
    }

    /**
     * Read data for features and outputs from a user-specified data file
     * The first line of the file must being with two integers specifying 
     * the number of columns containing features and outputs respectively.
     *
     * @param dataFileName: Absolute path to data file.
     * @param features: Array to store data for features.
     * @param outputs: Array to store data for outputs.
     */
    void readData(std::string dataFileName, 
		    std::vector<std::vector<double>> & features, 
		    std::vector<std::vector<double>> & outputs ) {
	    std::ifstream inp;
	    inp.open(dataFileName);
	    std::string line;
	    getline(inp, line);
	    std::stringstream ss(line);
	    int numberFeatures = 0;
	    int numberOutputs = 0;
	    ss >> numberFeatures >> numberOutputs ;

	    while (getline(inp, line)) {
		    ss << line;
		    std::vector<double> rowFeatures(numberFeatures, 0.);
		    std::vector<double> rowOutputs(numberOutputs, 0.);
		    for (int i = 0; i < numberFeatures; i++) ss >> rowFeatures[i] ;
		    for (int i = 0; i < numberOutputs; i++) ss >> rowOutputs[i];
		    features.push_back(rowFeatures);
		    outputs.push_back(rowOutputs);
	    }

    }


  };
} // namespace ml
