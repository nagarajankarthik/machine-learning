#include "utils/read_input.cpp"
#include "utils/logging.cpp"

using namespace std;

int main(int argc, char * argv [])
{
	string inputFileName = "input.json";
	if (argc > 1) inputFileName = argv[1];
	ReadInput inputReader (inputFileName);
	json params = inputReader.readJson(inputFileName);
    Logger logger("logfile.txt"); // Create logger instance

    // Example usage of the logger
    logger.log(INFO, "Program started.");
    logger.log(DEBUG, "Debugging information.");
    logger.log(ERROR, "An error occurred.");

    return 0;
}


