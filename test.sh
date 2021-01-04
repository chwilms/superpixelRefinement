python generateIntermediateResults.py 0 spxRefinedAttMask --init_weights spxRefinedAttMask-final.caffemodel --dataset val2017LVIS --end 5000
python generateFinalResults.py spxRefinedAttMask --dataset val2017LVIS --end 5000
python evalCOCONMS.py spxRefinedAttMask --dataset val2017LVIS --useSegm True --end 5000
