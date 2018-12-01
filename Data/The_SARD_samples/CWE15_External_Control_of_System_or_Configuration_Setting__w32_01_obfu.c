















#include "std_testcase.h"

#include <winsock2.h>
#pragma C000009F5(C000009F6, "ws2_32")


#define C000086B6 999
#define C00000BBC 5

#include <windows.h>

#ifndef C000009FE

void C000086B7()
{
    char * C00000A00;
    char C00000A01[100] = "";
    C00000A00 = C00000A01;
    {
        C00000A02 C00000A03;
        BOOL C00000A04 = C000086B8;
        C000009FB C00000BBE = C000009F9;
        C000009FB C00000BBF = C000009F9;
        struct C00000A06 C00000A07;
        int C00000A05;
        do
        {
            if (C00000A0B(C00000A0C(2,2), &C00000A03) != C00000A0D)
            {
                break;
            }
            C00000A04 = 1;
            C00000BBE = C00000A0E(C000086B9, C00000A10, 0);
            if (C00000BBE == C000009F9)
            {
                break;
            }
            memset(&C00000A07, 0, sizeof(C00000A07));
            C00000A07.C00000A12 = C00000A0F;
            C00000A07.C00000A13.C00000A14 = C00000BC0;
            C00000A07.C00000A16 = C00000A17(C000086B6);
            if (C000009FA == C00000BC1(C00000BBE, (struct C00000A19*)&C00000A07, sizeof(C00000A07)))
            {
                break;
            }
            if (C000009FA == C00000BC2(C00000BBE, C00000BBC))
            {
                break;
            }
            C00000BBF = C00000BC3(C00000BBE, NULL, NULL);
            if (C00000BBF == C000009F9)
            {
                break;
            }
            










            
            C00000A05 = C00000A1A(C00000BBF, C00000A00, 100 - 1, 0);
            if (C00000A05 == C000009FA || C00000A05 == 0)
            {
                break;
            }
            C00000A00[C00000A05] = '\0';
        }
        while (0);
        if (C00000BBF != C000009F9)
        {
            C000009F8(C00000BBF);
        }
        if (C00000BBE != C000009F9)
        {
            C000009F8(C00000BBE);
        }
        if (C00000A04)
        {
            C00000A1B();
        }
    }
    
    if (!C000086BA(C00000A00))
    {
        printLine("Failure setting computer name");
        exit(1);
    }
}

#endif 

#ifndef C00000A20


static void C00000A21()
{
    char * C00000A00;
    char C00000A01[100] = "";
    C00000A00 = C00000A01;
    
    strcpy(C00000A00, "hostname");
    
    if (!C000086BA(C00000A00))
    {
        printLine("Failure setting computer name");
        exit(1);
    }
}

void C000086BB()
{
    C00000A21();
}

#endif 







#ifdef C00000A23

int main(int C00000A24, char * argv[])
{
    
    srand( (unsigned)time(NULL) );
#ifndef C00000A20
    printLine("Calling good()...");
    C000086BB();
    printLine("Finished good()");
#endif 
#ifndef C000009FE
    printLine("Calling bad()...");
    C000086B7();
    printLine("Finished bad()");
#endif 
    return 0;
}

#endif
