import csv
import pyperclip

from fractions import Fraction

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
charges, dim, ops, lp, eon = [], [], [], [], []
with open('Q_reps_refined.csv', 'r') as file:
   table = csv.reader(file)
   next(table) # Skip the header row
   for row in table:
      charges.append([int(q) for q in row[:3]])
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
      
# s = "  \\toprule\n  \multicolumn{3}{c}{Rep} & $E/N$ & Min.\ $d$ & Ex.\ operators & \multicolumn{1}{c}{LP [GeV]} \\\\\n  \midrule\n"
d0 = 3
s = ""
for c,r,d,i,x in zip(charges,eon,dim,ops,lp):
   if (c[0] == 8 or c[0] == 27) and (c[2] < 0):
      continue
   if d != d0:
      s+= "  \midrule\n"
      d0 = d
   if r.denominator == 1:
      ratio = f"{r.numerator:d}"
   else:
      ratio = f"{r.numerator:d}/{r.denominator:d}"
   s+= f"  {c[0]} & {c[1]} & {c[2]} & {ratio} & {d} & {i} & {x:.1e} \\\\\n"
# s+= r"  \bottomrule"

print(s)
pyperclip.copy(s)