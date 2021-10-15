MAX_EPOCHS=13
SIZE_EPOCH=80000

while [[ $EPOCH -le $MAX_EPOCHS ]]
do
    echo "$EPOCH"
    echo 'start training'
    if [[ $EPOCH -eq 1 ]]
    then
        let STEP=EPOCH*SIZE_EPOCH
        python trainSpxRefinedAttMask.py 0 spxRefinedAttMask --init_weights resnet34.caffemodel --step 80000
    else
        let STEP_OLD=(EPOCH-1)*SIZE_EPOCH
        let STEP=EPOCH*SIZE_EPOCH
        python trainSpxRefinedAttMask.py 0 spxRefinedAttMask --restore spxRefinedAttMask_iter_$STEP_OLD.solverstate --step 80000
    fi
    let EPOCH=EPOCH+1
done
