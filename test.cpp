#include <stdio.h>
typedef struct Lijichu{
    int a;
    int *p;
}Lijichu;
int main()
{
    Lijichu *x;
    x->a=100;
    printf("%d",x->a);
    return 0;
}
