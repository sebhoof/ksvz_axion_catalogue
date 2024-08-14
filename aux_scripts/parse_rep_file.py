import csv
import numpy as np
import os
import pyperclip

from fractions import Fraction

dynkin_dict_2_times_2 = {1: 0, 2: 1, 3: 4, 4: 10, 5: 20, 6: 35, 7: 56, 8: 84}
dynkin_dict_3_times_2 = {1: 0, 3: 1, 6: 5, 8: 6, 10: 15, 15: 20, 152: 35, 21: 70, 24: 50, 27: 54, 28: 126, 35: 105, 36: 210, 42: 119, 45: 330, 48: 196, 55: 495, 60: 230}
def dynkin_times_2(rep: list[int]) -> list[int]:
    return [dynkin_dict_3_times_2[rep[0]], dynkin_dict_2_times_2[rep[1]], 2*rep[0]*rep[0]]

casimir_dict_2_times_12 = {1: 0, 2: 9, 3: 24, 4: 45, 5: 72, 6: 105, 7: 144, 8: 189}
casimir_dict_3_times_12 = {1: 0, 3: 16, 6: 40, 8: 36, 10: 72, 15: 64, 152: 112, 21: 160, 24: 100, 27: 96, 28: 216, 35: 144, 36: 280, 42: 136, 45: 352, 48: 196, 55: 432, 60: 184}
def casimir_times_12(rep: list[int]) -> list[int]:
    return [casimir_dict_3_times_12[rep[0]], casimir_dict_2_times_12[rep[1]], 12*rep[0]*rep[0]]

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
      info += dynkin_times_2(rep)
      info += casimir_times_12(rep)
      repinfo.append(info)
      
# Sort repinfo by dimension
repinfo = np.array(repinfo, dtype='int64')
repinfo = repinfo[repinfo[:,3].argsort()]
file_path = os.path.dirname(os.path.realpath(__file__))
header = "Data file for KSVZ reprensentations\nColumns: q_3 | q_2 | 6 * q_1 | min dim | 2 * Dyn_3 | 2 * Dyn_2 | 2 * Dyn_1 | 12 * Cas_3 | 12 * Cas_2 | 12 * Cas_1"
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