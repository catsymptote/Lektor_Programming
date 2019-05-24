# -*- coding: utf-8 -*-

import LSO

text = input("Encryption text:\t")
shift = int(input("Password (integer):\t"))

if shift < 1:
    shift = 1

print(LSO.LSO(text, shift))
