import csv
import numpy as np
import os
import pyperclip

from fractions import Fraction

dynkin_dict_2_times_36 = {1: 0, 2: 18, 3: 72, 4: 180, 5: 360, 6: 630, 7: 1008, 8: 1512}
dynkin_dict_3_times_36 = {1: 0, 3: 18, 6: 90, 8: 108, 10: 270, 15: 360, 152: 630, 21: 1260, 24: 900, 27: 972, 28: 2268, 35: 1890, 36: 3780, 42: 2142, 45: 5940, 48: 3528, 55: 8910, 60: 4140}
def dynkin_times_36(rep: list[int]) -> list[int]:
    return [dynkin_dict_3_times_36[rep[0]], dynkin_dict_2_times_36[rep[1]], rep[0]*rep[0]]

casimir_dict_2_times_36 = {1: 0, 2: 27, 3: 72, 4: 135, 5: 216, 6: 315, 7: 432, 8: 567}
casimir_dict_3_times_36 = {1: 0, 3: 48, 6: 120, 8: 108, 10: 216, 15: 192, 152: 336, 21: 480, 24: 300, 27: 288, 28: 648, 35: 432, 36: 840, 42: 408, 45: 1056, 48: 588, 55: 1296, 60: 552}
def casimir_times_36(rep: list[int]) -> list[int]:
    return [casimir_dict_3_times_36[rep[0]], casimir_dict_2_times_36[rep[1]], rep[0]*rep[0]]

# Dictionary of operators
operators = {
   'QLd': r"\bar{\newQ}_L", # conjugated LH heavy quark
   'QRd': r"\bar{\newQ}_R", # conjugated RH heavy quark
   'd': r"d_R", # RH down quark
   'u': r"u_R", # RH up quark
   'q': r"q_L", # LH quark doublet
   'e': r"e_R", # RH electron
   'l': r"\ell_L", # LH lepton doublet
   'dd': r"\bar{d}_R", # conjugated RH down quark
   'ud': r"\bar{u}_R", # conjugated RH up quark
   'qd': r"\bar{q}_L", # conjugated LH quark doublet
   'ed': r"\bar{e}_R", # conjugated RH electron
   'ld': r"\bar{\ell}_L", # conjugated LH lepton doublet
   'h': r"H", # Higgs
   'hd': r"H^\dagger", # conjugated Higgs
   'W': r"\sigma \cdot W", # SU(2) field strength tensor
   'G': r"\sigma \cdot G", # gluon field strength tensor
   'p': r"\slashed{\partial}" # momentum/derivative
}

# Open and read file to parse
charges, dim, ops, lp, eon, repinfo = [], [], [], [], [], []
with open('Q_reps_refined.csv', 'r') as file:
   table = csv.reader(file)
   next(table) # Skip the header row
   for row in table:
      rep = [int(q) for q in row[:3]]
      charges.append(rep)
      d = int(row[3])
      dim.append(d)
      examples = row[4:11]
      op_symbols = row[d+1].split('*')
      op_string = "$"
      for op in op_symbols:
         if '^' in op:
            op = op.split('^')
            op_string += r"("+operators[op[0]] + r")^{" + op[1] + r"}\,"
         else:
            try:
               fac = int(op)
               op_string += f"{fac:d}\,"
            except:
               op_string += operators[op] + r"\,"
      op_string = op_string[:-2]+"$"
      ops.append(op_string)
      lp.append(float(row[11]))
      e = Fraction(int(row[12]), int(row[13]))
      n = Fraction(int(row[14]), int(row[15]))
      eon.append(e/n)
      if (rep[0] == 8 or rep[0] == 27) and (rep[2] < 0):
         continue
      info = rep + [d]
      info += dynkin_times_36(rep)
      info += casimir_times_36(rep)
      repinfo.append(info)
      
# Sort repinfo by dimension
repinfo = np.array(repinfo, dtype='int64')
repinfo = repinfo[repinfo[:,3].argsort()]
file_path = os.path.dirname(os.path.realpath(__file__))
header = "Data file for KSVZ reprensentations\nColumns: q_3 | q_2 | 6 * q_1 | min dim | 36 * Dyn_3 | 36 * Dyn_2 | 36 * Dyn_1 | 36 * Cas_3 | 36 * Cas_2 | 36 * Cas_1"
np.savetxt(file_path+"/../ksvz_models/data/rep_info.dat", repinfo, fmt='%d', header=header, comments='#')

# Generate table and update rep dictionary
# s = "  \\toprule\n  \multicolumn{3}{c}{Rep} & $E/N$ & Min.\ $d$ & Ex.\ operators & \multicolumn{1}{c}{LP [GeV]} \\\\\n  \midrule\n"

new_i = 0
repdict = {}
replist = []
d0 = 3
s = ""
for c,r,d,i,x in zip(charges,eon,dim,ops,lp):
   if (c[0] == 8 or c[0] == 27) and (c[2] < 0):
      continue
   rep = [Fraction(c[0],1), Fraction(c[1],1), Fraction(c[2],6)]
   new_i += 1
   repdict[new_i] = [c[0],c[1],Fraction(c[2],6)]
   replist.append([c[0],c[1],c[2]])
   if d != d0:
      s+= "  \midrule\n"
      d0 = d
   if r.denominator == 1:
      ratio = f"{r.numerator:d}"
   else:
      ratio = f"{r.numerator:d}/{r.denominator:d}"
   s += f"  {c[0]} & {c[1]} & {c[2]} & {ratio} & {d} & {i} & {x:.1e} \\\\\n"
   
# s+= r"  \bottomrule"

print(s)
pyperclip.copy(s)
print("Repdict:")
print(repdict)
print("Replist:")
print(replist)