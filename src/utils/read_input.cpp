#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "json.hpp"
#include <iostream>

namespace ml
{
  /**
   * Reads input parameters from a file.
   * Only json file format is currently supported.
   */
  class ReadInput
  {
  public:
    std::ifstream inp;
    std::vector<int> integerInputFirst;

    ReadInput(std::string inputFileName) { inp.open(inputFileName); };
    ~ReadInput() { inp.close(); };

    nlohmann::json_abi_v3_11_3::json readJson()
    {

      std::stringstream buffer;
      buffer << inp.rdbuf();
      auto inputParameters = nlohmann::json::parse(buffer.str());
      return inputParameters;
    }

    // trim from start (in place)
    inline void ltrim(std::string &s)
    {
      s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
                                      { return !std::isspace(ch); }));
    }

    // trim from end (in place)
    inline void rtrim(std::string &s)
    {
      s.erase(std::find_if(s.rbegin(), s.rend(),
                           [](unsigned char ch)
                           { return !std::isspace(ch); })
                  .base(),
              s.end());
    }

    // trim from both ends (in place)
    inline void trim(std::string &s)
    {
      rtrim(s);
      ltrim(s);
    }

    std::unordered_map<std::string, std::string> readOld()
    {
      std::string line = "";
      std::unordered_map<std::string, std::string> params{};

      while (getline(inp, line))
      {
        trim(line);
        if (line[0] == '#' || line.size() == 0)
          continue;

        // find key value pair from file
        size_t stopPosition = line.find(':');
        std::string keyName = line.substr(0, stopPosition);

        // assign value if one is present
        std::string valueName = "";
        if (stopPosition + 2 < line.length())
          valueName = line.substr(stopPosition + 2);

        if (params.find(keyName) == params.end())
          params[keyName] = valueName;
      }
      return params;
    }
  };
} // namespace ml
