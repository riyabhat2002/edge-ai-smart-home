#include <iostream>
#include <string>
#include "llama.h"

int main()
{
    llama_backend_init();
    std::cout << "Llama backend initialized successfully." << std::endl;

    llama_backend_free();
    std::cout << "Llama backend freed successfully." << std::endl;
    
    return 0;
}