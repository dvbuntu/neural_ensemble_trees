import pandas as pd
import warnings
import pickle as pk
from glob import glob
import argparse
parser = argparse.ArgumentParser(description='Make table of output')
parser.add_argument('-f','--file', action='append',
                        default=[],
                        help='Read this output file and add as row to table')
parser.add_argument('-o','--outfile',
                        default='',
                        help='Output file to save final results in big csv')

args = parser.parse_args()

files = args.file

df = pd.DataFrame(columns=['Dataset', 'Forest', 'NN2',
                            'NF2 Fresh', 'NF2 Weights',
                            'NF2 Sparse Fresh', 'NF2 Sparse Weights'
                            ])
#                            'NN2 BART',
#                            'NBART2 Fresh', 'NBART2 Weights',
#                            'NBART2 Sparse Fresh', 'NBART2 Sparse Weights',
#                            'BART'])

datasets = set()
rows = []
for base in files:
    for outf in glob(base + '_*.out'):
        try:
            with open(outf, 'rb') as f:
                res = pk.load(f)
        except FileNotFoundError:
            warnings.warn(f"Couldn't find {outf}")
            continue
        start = outf.index('_')
        end = outf.index('.')
        i = outf[start+1:end]
        row = {}
        name = list(res.keys())[0][0]
        datasets.add(name)
        row['Dataset']  = name + ' RF'
        row['Forest'] = res[(name, 'randomforest')]['randomforest']
        row['NN2'] = res[(name, 'randomforest')]['NN2']
        row['NF2 Fresh'] = res[(name, 'randomforest')]['NRF2 full no weights']
        row['NF2 Weights'] = res[(name, 'randomforest')]['NRF2 full']
        row['NF2 Sparse Fresh'] = res[(name, 'randomforest')]['NRF2 sparse no weights']
        row['NF2 Sparse Weights'] = res[(name, 'randomforest')]['NRF2 sparse']

        rows.append(row)

        row2 = {}
        row2['Dataset']  = name + ' BART'
        row2['Forest'] = res[(name, 'bart')]['bart']
        row2['NN2'] = res[(name, 'bart')]['NN2']
        row2['NF2 Fresh'] = res[(name, 'bart')]['NRF2 full no weights']
        row2['NF2 Weights'] = res[(name, 'bart')]['NRF2 full']
        row2['NF2 Sparse Fresh'] = res[(name, 'bart')]['NRF2 sparse no weights']
        row2['NF2 Sparse Weights'] = res[(name, 'bart')]['NRF2 sparse']
        rows.append(row2)

df = pd.concat([df, pd.DataFrame(rows)], sort=False)

df_out = pd.DataFrame(columns=['Dataset', 'Forest', 'NN2',
                            'NF2 Fresh', 'NF2 Weights',
                            'NF2 Sparse Fresh', 'NF2 Sparse Weights'
                            ])

formatted = []
for name in sorted(set(df['Dataset'])):
    M = df[df['Dataset'] == name].mean()
    S = df[df['Dataset'] == name].std()
    formatted.append([name] + [f'{m:6.3f} ({s:6.3f})' for m,s in zip(M,S)])


tmp = pd.DataFrame(formatted)
new_cols = {x: y for x, y in zip(tmp.columns, df_out.columns)}

df_out = pd.concat([df_out, tmp.rename(columns=new_cols)], sort=False)

print(df_out.to_markdown(index=False))

if args.outfile:
    with open(args.outfile, 'w') as f:
        print(df.to_csv(index=False), file=f)
