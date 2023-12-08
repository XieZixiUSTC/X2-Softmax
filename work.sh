# python verification.py --resultroot 'D:\30.python\ReluSoftmax\OUTPUT\20231203 X2SOFTMAX' --modelstep 800000 802000 804000 806000 808000

# python evaluate_IJB.py --model_prefix 'model_step808000.pt' --result_dir 'D:\30.python\ReluSoftmax\OUTPUT\20231203 X2SOFTMAX' --target 'IJBB'
# python evaluate_IJB.py --model_prefix 'model_step808000.pt' --result_dir 'D:\30.python\ReluSoftmax\OUTPUT\20231203 X2SOFTMAX' --target 'IJBC'

# python train.py --loss 'X2SOFTMAX' --lossparameters 64 -0.9 -0.0 0.65 --output '.\OUTPUT\20231203 X2SOFTMAX'

echo press any key to continue
read -n 1
