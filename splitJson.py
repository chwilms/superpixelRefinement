'''
@author Christian Wilms
@date 01/05/21
'''

import json
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

with open('spxGT_train2014_FH.json') as f:
    data=json.load(f)
    
for scale in [8,16,24,32,48,64,96,128]:
    resultDict = []
    for k in data.keys()[:]:
        resultDict.append((int(k),data[k][str(scale)]))
    resultDict=dict(resultDict)    
        
    with io.open('spxGT_train2014_FH_'+str(scale)+'.json', 'w', encoding='utf-8') as f:
        str_ = json.dumps(resultDict8, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        f.write(to_unicode(str_))
    
