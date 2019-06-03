# -*- coding: utf-8 -*-

import LSO

text = input("Decryption text:\t")
shift = int(input("Password (integer):\t"))

if shift < 1:
    shift = 1

print(LSO.RSO(text, shift).title())
