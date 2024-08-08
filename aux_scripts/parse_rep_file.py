import csv
import pyperclip

from fractions import Fraction

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
      ops.append(row[d+1])
      lp.append(float(row[11]))
      e = Fraction(int(row[12]), int(row[13]))
      n = Fraction(int(row[14]), int(row[15]))
      eon.append(e/n)

# Produce a LaTeX table and copy to clipboard
# s = "  \\toprule\n  \multicolumn{3}{c}{Rep} & $E/N$ & Min.\ $d$ & Ex.\ operators & \multicolumn{1}{c}{LP [GeV]} \\\\\n  \midrule\n"
s = ""
for c,r,d,i,x in zip(charges,eon,dim,ops,lp):
   s+= f"  ${c[0]}$ & ${c[1]}$ & ${c[2]}$ & {r.numerator:d}/{r.denominator:d} & ${d}$ & ${i}$ & {x:.2e} \\\\\n"
# s+= r"  \bottomrule"

print(s)
pyperclip.copy(s)
