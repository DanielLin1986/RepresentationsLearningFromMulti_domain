#include "std_testcase.h"

#include <winsock2.h>
#pragma comment(lib, "ws2_32")


#define LISTEN_PORT 999
#define LISTEN_BACKLOG 5

#include <windows.h>

#ifndef OMITBAD

void CWE15_External_Control_of_System_or_Configuration_Setting__w32_01_bad()
{
    char * data;
    char dataBuffer[100] = "";
    data = dataBuffer;
    {
        WSADATA wsaData;
        BOOL wsaDataInit = FALSE;
        SOCKET listenSocket = INVALID_SOCKET;
        SOCKET acceptSocket = INVALID_SOCKET;
        struct sockaddr_in service;
        int recvResult;
        do
        {
            if (WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR)
            {
                break;
            }
            wsaDataInit = 1;
            listenSocket = socket(PF_INET, SOCK_STREAM, 0);
            if (listenSocket == INVALID_SOCKET)
            {
                break;
            }
            memset(&service, 0, sizeof(service));
            service.sin_family = AF_INET;
            service.sin_addr.s_addr = INADDR_ANY;
            service.sin_port = htons(LISTEN_PORT);
            if (SOCKET_ERROR == bind(listenSocket, (struct sockaddr*)&service, sizeof(service)))
            {
                break;
            }
            if (SOCKET_ERROR == listen(listenSocket, LISTEN_BACKLOG))
            {
                break;
            }
            acceptSocket = accept(listenSocket, NULL, NULL);
            if (acceptSocket == INVALID_SOCKET)
            {
                break;
            }           
            recvResult = recv(acceptSocket, data, 100 - 1, 0);
            if (recvResult == SOCKET_ERROR || recvResult == 0)
            {
                break;
            }
            data[recvResult] = '\0';
        }
        while (0);
        if (acceptSocket != INVALID_SOCKET)
        {
            closesocket(acceptSocket);
        }
        if (listenSocket != INVALID_SOCKET)
        {
            closesocket(listenSocket);
        }
        if (wsaDataInit)
        {
            WSACleanup();
        }
    }
    
    if (!SetComputerNameA(data))
    {
        printLine("Failure setting computer name");
        exit(1);
    }
}

#endif 

#ifndef OMITGOOD


static void goodG2B()
{
    char * data;
    char dataBuffer[100] = "";
    data = dataBuffer;
    
    strcpy(data, "hostname");
    
    if (!SetComputerNameA(data))
    {
        printLine("Failure setting computer name");
        exit(1);
    }
}

void CWE15_External_Control_of_System_or_Configuration_Setting__w32_01_good()
{
    goodG2B();
}

#endif 

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE15_External_Control_of_System_or_Configuration_Setting__w32_01_good();
    printLine("Finished good()");
#endif 
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE15_External_Control_of_System_or_Configuration_Setting__w32_01_bad();
    printLine("Finished bad()");
#endif 
    return 0;
}

#endif
