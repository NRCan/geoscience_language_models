#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "common.h"

int main() {
  CREC cr;
  FILE * fp;
  fp = fopen("/var/tmp/tmp.4.gb8s6o", "rb");
//  /var/tmp/tmp.6.nK6Z45
//  /var/tmp/tmp.0.8e4onp
//  /var/tmp/tmp.0.QV9XsH
//  /var/tmp/tmp.3.yKmzTm
//  /var/tmp/tmp.5.ApwEEj
//  /var/tmp/tmp.0.qE3hhW
//  /var/tmp/tmp.0.k5jipJ
//  /var/tmp/tmp.4.gb8s6o

  while (1) {
    fread(&cr, sizeof(CREC), 1, fp);
    if(feof(fp)) {
      break;
    }

    printf("\nword1: %d", cr.word1);
    printf("\nword2: %d", cr.word2);
    printf("\nval: %f", cr.val);
    printf("\n");
  }
  printf("\nFinished\n");

  fclose(fp);
  return 0;
}