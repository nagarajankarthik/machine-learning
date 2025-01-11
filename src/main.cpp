#include "utils/read_input.cpp"
#include "utils/logging.cpp"
#include <string>
using namespace ml;
using namespace std;

int main(int argc, char *argv[])
{
    string inputFileName = "input.json";
    if (argc > 1)
        inputFileName = argv[1];
    ReadInput inputReader(inputFileName);
    auto inputParameters = inputReader.readJson();

    string logFileName = inputParameters["general"]["logfile"];

    // Create logger instance
    Logger logger("../logs/" + logFileName);

    // Example usage of the logger
    logger.log(INFO, "Program started.");
    logger.log(DEBUG, "Debugging information.");
    logger.log(ERROR, "An error occurred.");

    return 0;
}
