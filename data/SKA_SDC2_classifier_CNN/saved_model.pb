ψη
Ρ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Η

B_conv3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameB_conv3_1/kernel

$B_conv3_1/kernel/Read/ReadVariableOpReadVariableOpB_conv3_1/kernel**
_output_shapes
: *
dtype0
t
B_conv3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameB_conv3_1/bias
m
"B_conv3_1/bias/Read/ReadVariableOpReadVariableOpB_conv3_1/bias*
_output_shapes
: *
dtype0

B_conv3_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameB_conv3_2/kernel

$B_conv3_2/kernel/Read/ReadVariableOpReadVariableOpB_conv3_2/kernel**
_output_shapes
:  *
dtype0
t
B_conv3_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameB_conv3_2/bias
m
"B_conv3_2/bias/Read/ReadVariableOpReadVariableOpB_conv3_2/bias*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ϊ"
valueΠ"BΝ" BΖ"
υ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
R
!trainable_variables
"regularization_losses
#	variables
$	keras_api
R
%trainable_variables
&regularization_losses
'	variables
(	keras_api
h

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
h

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
 
F
0
1
2
3
)4
*5
36
47
98
:9
 
F
0
1
2
3
)4
*5
36
47
98
:9
­
trainable_variables

?layers
@non_trainable_variables
regularization_losses
Alayer_metrics
	variables
Blayer_regularization_losses
Cmetrics
 
\Z
VARIABLE_VALUEB_conv3_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEB_conv3_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Dlayers
trainable_variables
Enon_trainable_variables
Flayer_metrics
regularization_losses
	variables
Glayer_regularization_losses
Hmetrics
 
 
 
­

Ilayers
trainable_variables
Jnon_trainable_variables
Klayer_metrics
regularization_losses
	variables
Llayer_regularization_losses
Mmetrics
\Z
VARIABLE_VALUEB_conv3_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEB_conv3_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Nlayers
trainable_variables
Onon_trainable_variables
Player_metrics
regularization_losses
	variables
Qlayer_regularization_losses
Rmetrics
 
 
 
­

Slayers
!trainable_variables
Tnon_trainable_variables
Ulayer_metrics
"regularization_losses
#	variables
Vlayer_regularization_losses
Wmetrics
 
 
 
­

Xlayers
%trainable_variables
Ynon_trainable_variables
Zlayer_metrics
&regularization_losses
'	variables
[layer_regularization_losses
\metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
­

]layers
+trainable_variables
^non_trainable_variables
_layer_metrics
,regularization_losses
-	variables
`layer_regularization_losses
ametrics
 
 
 
­

blayers
/trainable_variables
cnon_trainable_variables
dlayer_metrics
0regularization_losses
1	variables
elayer_regularization_losses
fmetrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
­

glayers
5trainable_variables
hnon_trainable_variables
ilayer_metrics
6regularization_losses
7	variables
jlayer_regularization_losses
kmetrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
­

llayers
;trainable_variables
mnon_trainable_variables
nlayer_metrics
<regularization_losses
=	variables
olayer_regularization_losses
pmetrics
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*4
_output_shapes"
 :?????????Θ*
dtype0*)
shape :?????????Θ
θ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1B_conv3_1/kernelB_conv3_1/biasB_conv3_2/kernelB_conv3_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *,
f'R%
#__inference_signature_wrapper_28487
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$B_conv3_1/kernel/Read/ReadVariableOp"B_conv3_1/bias/Read/ReadVariableOp$B_conv3_2/kernel/Read/ReadVariableOp"B_conv3_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *'
f"R 
__inference__traced_save_28823
Β
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameB_conv3_1/kernelB_conv3_1/biasB_conv3_2/kernelB_conv3_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 **
f%R#
!__inference__traced_restore_28863η
§	
¬
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_28663

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# *
paddingVALID*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????# 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????' :::[ W
3
_output_shapes!
:?????????' 
 
_user_specified_nameinputs
§
ͺ
B__inference_dense_1_layer_call_and_return_conditional_losses_28741

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs

~
)__inference_B_conv3_2_layer_call_fn_28672

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????# *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_281672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????' ::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????' 
 
_user_specified_nameinputs
ώ5
§
G__inference_functional_1_layer_call_and_return_conditional_losses_28538

inputs,
(b_conv3_1_conv3d_readvariableop_resource-
)b_conv3_1_biasadd_readvariableop_resource,
(b_conv3_2_conv3d_readvariableop_resource-
)b_conv3_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity·
B_conv3_1/Conv3D/ReadVariableOpReadVariableOp(b_conv3_1_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02!
B_conv3_1/Conv3D/ReadVariableOpΘ
B_conv3_1/Conv3DConv3Dinputs'B_conv3_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ *
paddingVALID*
strides	
2
B_conv3_1/Conv3Dͺ
 B_conv3_1/BiasAdd/ReadVariableOpReadVariableOp)b_conv3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 B_conv3_1/BiasAdd/ReadVariableOp΅
B_conv3_1/BiasAddBiasAddB_conv3_1/Conv3D:output:0(B_conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ 2
B_conv3_1/BiasAdd
B_conv3_1/ReluReluB_conv3_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????Δ 2
B_conv3_1/ReluΡ
B_pool2_1/MaxPool3D	MaxPool3DB_conv3_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????' *
ksize	
*
paddingVALID*
strides	
2
B_pool2_1/MaxPool3D·
B_conv3_2/Conv3D/ReadVariableOpReadVariableOp(b_conv3_2_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
B_conv3_2/Conv3D/ReadVariableOpέ
B_conv3_2/Conv3DConv3DB_pool2_1/MaxPool3D:output:0'B_conv3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# *
paddingVALID*
strides	
2
B_conv3_2/Conv3Dͺ
 B_conv3_2/BiasAdd/ReadVariableOpReadVariableOp)b_conv3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 B_conv3_2/BiasAdd/ReadVariableOp΄
B_conv3_2/BiasAddBiasAddB_conv3_2/Conv3D:output:0(B_conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# 2
B_conv3_2/BiasAdd
B_conv3_2/ReluReluB_conv3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????# 2
B_conv3_2/ReluΡ
B_pool2_2/MaxPool3D	MaxPool3DB_conv3_2/Relu:activations:0*
T0*3
_output_shapes!
:????????? *
ksize	
*
paddingVALID*
strides	
2
B_pool2_2/MaxPool3Dk
B_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
B_out/Const
B_out/ReshapeReshapeB_pool2_2/MaxPool3D:output:0B_out/Const:output:0*
T0*(
_output_shapes
:?????????2
B_out/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulB_out/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeΜ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2 
dropout/dropout/GreaterEqual/yή
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mul_1₯
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul€
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp‘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Relu₯
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul€
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp‘
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoidg
IdentityIdentitydense_2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ:::::::::::\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
ά
E
)__inference_B_pool2_1_layer_call_fn_28112

inputs
identityϋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_281062
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
ΐ
\
@__inference_B_out_layer_call_and_return_conditional_losses_28190

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? :[ W
3
_output_shapes!
:????????? 
 
_user_specified_nameinputs
α
|
'__inference_dense_2_layer_call_fn_28770

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallϋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_282932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ύ(
χ
G__inference_functional_1_layer_call_and_return_conditional_losses_28310
input_1
b_conv3_1_28150
b_conv3_1_28152
b_conv3_2_28178
b_conv3_2_28180
dense_28220
dense_28222
dense_1_28277
dense_1_28279
dense_2_28304
dense_2_28306
identity’!B_conv3_1/StatefulPartitionedCall’!B_conv3_2/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’dropout/StatefulPartitionedCall­
!B_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinput_1b_conv3_1_28150b_conv3_1_28152*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????Δ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_281392#
!B_conv3_1/StatefulPartitionedCall
B_pool2_1/PartitionedCallPartitionedCall*B_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????' * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_281062
B_pool2_1/PartitionedCallΗ
!B_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"B_pool2_1/PartitionedCall:output:0b_conv3_2_28178b_conv3_2_28180*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????# *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_281672#
!B_conv3_2/StatefulPartitionedCall
B_pool2_2/PartitionedCallPartitionedCall*B_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_281182
B_pool2_2/PartitionedCallπ
B_out/PartitionedCallPartitionedCall"B_pool2_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_B_out_layer_call_and_return_conditional_losses_281902
B_out/PartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCallB_out/PartitionedCall:output:0dense_28220dense_28222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_282092
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_282372!
dropout/StatefulPartitionedCall·
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_28277dense_1_28279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_282662!
dense_1/StatefulPartitionedCall·
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_28304dense_2_28306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_282932!
dense_2/StatefulPartitionedCallΚ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^B_conv3_1/StatefulPartitionedCall"^B_conv3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::2F
!B_conv3_1/StatefulPartitionedCall!B_conv3_1/StatefulPartitionedCall2F
!B_conv3_2/StatefulPartitionedCall!B_conv3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????Θ
!
_user_specified_name	input_1
	
ϊ
,__inference_functional_1_layer_call_fn_28402
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_283792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????Θ
!
_user_specified_name	input_1
Ε
`
B__inference_dropout_layer_call_and_return_conditional_losses_28720

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
­	
¬
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_28643

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOpͺ
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ *
paddingVALID*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ 2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :?????????Δ 2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :?????????Δ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????Θ:::\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
ΐ
\
@__inference_B_out_layer_call_and_return_conditional_losses_28678

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? :[ W
3
_output_shapes!
:????????? 
 
_user_specified_nameinputs
ά
E
)__inference_B_pool2_2_layer_call_fn_28124

inputs
identityϋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_281182
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
©
ͺ
B__inference_dense_2_layer_call_and_return_conditional_losses_28293

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
­	
¬
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_28139

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOpͺ
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ *
paddingVALID*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ 2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :?????????Δ 2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :?????????Δ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????Θ:::\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
Ε
`
B__inference_dropout_layer_call_and_return_conditional_losses_28242

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
ϊ(
φ
G__inference_functional_1_layer_call_and_return_conditional_losses_28379

inputs
b_conv3_1_28349
b_conv3_1_28351
b_conv3_2_28355
b_conv3_2_28357
dense_28362
dense_28364
dense_1_28368
dense_1_28370
dense_2_28373
dense_2_28375
identity’!B_conv3_1/StatefulPartitionedCall’!B_conv3_2/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’dropout/StatefulPartitionedCall¬
!B_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinputsb_conv3_1_28349b_conv3_1_28351*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????Δ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_281392#
!B_conv3_1/StatefulPartitionedCall
B_pool2_1/PartitionedCallPartitionedCall*B_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????' * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_281062
B_pool2_1/PartitionedCallΗ
!B_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"B_pool2_1/PartitionedCall:output:0b_conv3_2_28355b_conv3_2_28357*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????# *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_281672#
!B_conv3_2/StatefulPartitionedCall
B_pool2_2/PartitionedCallPartitionedCall*B_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_281182
B_pool2_2/PartitionedCallπ
B_out/PartitionedCallPartitionedCall"B_pool2_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_B_out_layer_call_and_return_conditional_losses_281902
B_out/PartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCallB_out/PartitionedCall:output:0dense_28362dense_28364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_282092
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_282372!
dropout/StatefulPartitionedCall·
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_28368dense_1_28370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_282662!
dense_1/StatefulPartitionedCall·
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_28373dense_2_28375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_282932!
dense_2/StatefulPartitionedCallΚ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^B_conv3_1/StatefulPartitionedCall"^B_conv3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::2F
!B_conv3_1/StatefulPartitionedCall!B_conv3_1/StatefulPartitionedCall2F
!B_conv3_2/StatefulPartitionedCall!B_conv3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
χ7

 __inference__wrapped_model_28100
input_19
5functional_1_b_conv3_1_conv3d_readvariableop_resource:
6functional_1_b_conv3_1_biasadd_readvariableop_resource9
5functional_1_b_conv3_2_conv3d_readvariableop_resource:
6functional_1_b_conv3_2_biasadd_readvariableop_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource7
3functional_1_dense_1_matmul_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource7
3functional_1_dense_2_matmul_readvariableop_resource8
4functional_1_dense_2_biasadd_readvariableop_resource
identityή
,functional_1/B_conv3_1/Conv3D/ReadVariableOpReadVariableOp5functional_1_b_conv3_1_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02.
,functional_1/B_conv3_1/Conv3D/ReadVariableOpπ
functional_1/B_conv3_1/Conv3DConv3Dinput_14functional_1/B_conv3_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ *
paddingVALID*
strides	
2
functional_1/B_conv3_1/Conv3DΡ
-functional_1/B_conv3_1/BiasAdd/ReadVariableOpReadVariableOp6functional_1_b_conv3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-functional_1/B_conv3_1/BiasAdd/ReadVariableOpι
functional_1/B_conv3_1/BiasAddBiasAdd&functional_1/B_conv3_1/Conv3D:output:05functional_1/B_conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ 2 
functional_1/B_conv3_1/BiasAddͺ
functional_1/B_conv3_1/ReluRelu'functional_1/B_conv3_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????Δ 2
functional_1/B_conv3_1/Reluψ
 functional_1/B_pool2_1/MaxPool3D	MaxPool3D)functional_1/B_conv3_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????' *
ksize	
*
paddingVALID*
strides	
2"
 functional_1/B_pool2_1/MaxPool3Dή
,functional_1/B_conv3_2/Conv3D/ReadVariableOpReadVariableOp5functional_1_b_conv3_2_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02.
,functional_1/B_conv3_2/Conv3D/ReadVariableOp
functional_1/B_conv3_2/Conv3DConv3D)functional_1/B_pool2_1/MaxPool3D:output:04functional_1/B_conv3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# *
paddingVALID*
strides	
2
functional_1/B_conv3_2/Conv3DΡ
-functional_1/B_conv3_2/BiasAdd/ReadVariableOpReadVariableOp6functional_1_b_conv3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-functional_1/B_conv3_2/BiasAdd/ReadVariableOpθ
functional_1/B_conv3_2/BiasAddBiasAdd&functional_1/B_conv3_2/Conv3D:output:05functional_1/B_conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# 2 
functional_1/B_conv3_2/BiasAdd©
functional_1/B_conv3_2/ReluRelu'functional_1/B_conv3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????# 2
functional_1/B_conv3_2/Reluψ
 functional_1/B_pool2_2/MaxPool3D	MaxPool3D)functional_1/B_conv3_2/Relu:activations:0*
T0*3
_output_shapes!
:????????? *
ksize	
*
paddingVALID*
strides	
2"
 functional_1/B_pool2_2/MaxPool3D
functional_1/B_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
functional_1/B_out/ConstΔ
functional_1/B_out/ReshapeReshape)functional_1/B_pool2_2/MaxPool3D:output:0!functional_1/B_out/Const:output:0*
T0*(
_output_shapes
:?????????2
functional_1/B_out/ReshapeΗ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpΙ
functional_1/dense/MatMulMatMul#functional_1/B_out/Reshape:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
functional_1/dense/MatMulΕ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpΝ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
functional_1/dense/BiasAdd
functional_1/dense/ReluRelu#functional_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
functional_1/dense/Relu£
functional_1/dropout/IdentityIdentity%functional_1/dense/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
functional_1/dropout/IdentityΜ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOp?
functional_1/dense_1/MatMulMatMul&functional_1/dropout/Identity:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense_1/MatMulΛ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpΥ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense_1/BiasAdd
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
functional_1/dense_1/ReluΜ
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpΣ
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense_2/MatMulΛ
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpΥ
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense_2/BiasAdd 
functional_1/dense_2/SigmoidSigmoid%functional_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
functional_1/dense_2/Sigmoidt
IdentityIdentity functional_1/dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ:::::::::::] Y
4
_output_shapes"
 :?????????Θ
!
_user_specified_name	input_1
	
ω
,__inference_functional_1_layer_call_fn_28632

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCallθ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_284372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
§
ͺ
B__inference_dense_1_layer_call_and_return_conditional_losses_28266

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ω
ρ
#__inference_signature_wrapper_28487
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCallΒ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *)
f$R"
 __inference__wrapped_model_281002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????Θ
!
_user_specified_name	input_1
¨
¨
@__inference_dense_layer_call_and_return_conditional_losses_28694

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
§	
¬
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_28167

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# *
paddingVALID*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????# 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????' :::[ W
3
_output_shapes!
:?????????' 
 
_user_specified_nameinputs
ώ

a
B__inference_dropout_layer_call_and_return_conditional_losses_28715

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΄
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΎ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
Υ'
Υ
G__inference_functional_1_layer_call_and_return_conditional_losses_28343
input_1
b_conv3_1_28313
b_conv3_1_28315
b_conv3_2_28319
b_conv3_2_28321
dense_28326
dense_28328
dense_1_28332
dense_1_28334
dense_2_28337
dense_2_28339
identity’!B_conv3_1/StatefulPartitionedCall’!B_conv3_2/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall­
!B_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinput_1b_conv3_1_28313b_conv3_1_28315*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????Δ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_281392#
!B_conv3_1/StatefulPartitionedCall
B_pool2_1/PartitionedCallPartitionedCall*B_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????' * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_281062
B_pool2_1/PartitionedCallΗ
!B_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"B_pool2_1/PartitionedCall:output:0b_conv3_2_28319b_conv3_2_28321*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????# *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_281672#
!B_conv3_2/StatefulPartitionedCall
B_pool2_2/PartitionedCallPartitionedCall*B_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_281182
B_pool2_2/PartitionedCallπ
B_out/PartitionedCallPartitionedCall"B_pool2_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_B_out_layer_call_and_return_conditional_losses_281902
B_out/PartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCallB_out/PartitionedCall:output:0dense_28326dense_28328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_282092
dense/StatefulPartitionedCallω
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_282422
dropout/PartitionedCall―
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_28332dense_1_28334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_282662!
dense_1/StatefulPartitionedCall·
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_28337dense_2_28339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_282932!
dense_2/StatefulPartitionedCall¨
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^B_conv3_1/StatefulPartitionedCall"^B_conv3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::2F
!B_conv3_1/StatefulPartitionedCall!B_conv3_1/StatefulPartitionedCall2F
!B_conv3_2/StatefulPartitionedCall!B_conv3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????Θ
!
_user_specified_name	input_1
Α
`
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_28106

inputs
identityΛ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
£
`
'__inference_dropout_layer_call_fn_28725

inputs
identity’StatefulPartitionedCallα
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_282372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
°-

!__inference__traced_restore_28863
file_prefix%
!assignvariableop_b_conv3_1_kernel%
!assignvariableop_1_b_conv3_1_bias'
#assignvariableop_2_b_conv3_2_kernel%
!assignvariableop_3_b_conv3_2_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias%
!assignvariableop_8_dense_2_kernel#
assignvariableop_9_dense_2_bias
identity_11’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9Ν
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ω
valueΟBΜB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names€
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesβ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_b_conv3_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_b_conv3_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_b_conv3_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_b_conv3_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4€
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5’
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¦
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7€
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¦
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9€
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpΊ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
­
A
%__inference_B_out_layer_call_fn_28683

inputs
identityΘ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_B_out_layer_call_and_return_conditional_losses_281902
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? :[ W
3
_output_shapes!
:????????? 
 
_user_specified_nameinputs
-
§
G__inference_functional_1_layer_call_and_return_conditional_losses_28582

inputs,
(b_conv3_1_conv3d_readvariableop_resource-
)b_conv3_1_biasadd_readvariableop_resource,
(b_conv3_2_conv3d_readvariableop_resource-
)b_conv3_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity·
B_conv3_1/Conv3D/ReadVariableOpReadVariableOp(b_conv3_1_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02!
B_conv3_1/Conv3D/ReadVariableOpΘ
B_conv3_1/Conv3DConv3Dinputs'B_conv3_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ *
paddingVALID*
strides	
2
B_conv3_1/Conv3Dͺ
 B_conv3_1/BiasAdd/ReadVariableOpReadVariableOp)b_conv3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 B_conv3_1/BiasAdd/ReadVariableOp΅
B_conv3_1/BiasAddBiasAddB_conv3_1/Conv3D:output:0(B_conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????Δ 2
B_conv3_1/BiasAdd
B_conv3_1/ReluReluB_conv3_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????Δ 2
B_conv3_1/ReluΡ
B_pool2_1/MaxPool3D	MaxPool3DB_conv3_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????' *
ksize	
*
paddingVALID*
strides	
2
B_pool2_1/MaxPool3D·
B_conv3_2/Conv3D/ReadVariableOpReadVariableOp(b_conv3_2_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
B_conv3_2/Conv3D/ReadVariableOpέ
B_conv3_2/Conv3DConv3DB_pool2_1/MaxPool3D:output:0'B_conv3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# *
paddingVALID*
strides	
2
B_conv3_2/Conv3Dͺ
 B_conv3_2/BiasAdd/ReadVariableOpReadVariableOp)b_conv3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 B_conv3_2/BiasAdd/ReadVariableOp΄
B_conv3_2/BiasAddBiasAddB_conv3_2/Conv3D:output:0(B_conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????# 2
B_conv3_2/BiasAdd
B_conv3_2/ReluReluB_conv3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????# 2
B_conv3_2/ReluΡ
B_pool2_2/MaxPool3D	MaxPool3DB_conv3_2/Relu:activations:0*
T0*3
_output_shapes!
:????????? *
ksize	
*
paddingVALID*
strides	
2
B_pool2_2/MaxPool3Dk
B_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
B_out/Const
B_out/ReshapeReshapeB_pool2_2/MaxPool3D:output:0B_out/Const:output:0*
T0*(
_output_shapes
:?????????2
B_out/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulB_out/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout/Identity₯
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul€
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp‘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Relu₯
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul€
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp‘
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoidg
IdentityIdentitydense_2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ:::::::::::\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
	
ω
,__inference_functional_1_layer_call_fn_28607

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCallθ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_283792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
¨
¨
@__inference_dense_layer_call_and_return_conditional_losses_28209

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

~
)__inference_B_conv3_1_layer_call_fn_28652

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????Δ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_281392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????Δ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????Θ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
©
ͺ
B__inference_dense_2_layer_call_and_return_conditional_losses_28761

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ί
z
%__inference_dense_layer_call_fn_28703

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallω
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_282092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
‘"
»
__inference__traced_save_28823
file_prefix/
+savev2_b_conv3_1_kernel_read_readvariableop-
)savev2_b_conv3_1_bias_read_readvariableop/
+savev2_b_conv3_2_kernel_read_readvariableop-
)savev2_b_conv3_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7048e62f560e4b988bc377491f51c07a/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameΗ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ω
valueΟBΜB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesμ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_b_conv3_1_kernel_read_readvariableop)savev2_b_conv3_1_bias_read_readvariableop+savev2_b_conv3_2_kernel_read_readvariableop)savev2_b_conv3_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapeso
m: : : :  : :	@:@:@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: 
Α
`
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_28118

inputs
identityΛ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs

C
'__inference_dropout_layer_call_fn_28730

inputs
identityΙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_282422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?'
Τ
G__inference_functional_1_layer_call_and_return_conditional_losses_28437

inputs
b_conv3_1_28407
b_conv3_1_28409
b_conv3_2_28413
b_conv3_2_28415
dense_28420
dense_28422
dense_1_28426
dense_1_28428
dense_2_28431
dense_2_28433
identity’!B_conv3_1/StatefulPartitionedCall’!B_conv3_2/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall¬
!B_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinputsb_conv3_1_28407b_conv3_1_28409*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????Δ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_281392#
!B_conv3_1/StatefulPartitionedCall
B_pool2_1/PartitionedCallPartitionedCall*B_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????' * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_281062
B_pool2_1/PartitionedCallΗ
!B_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"B_pool2_1/PartitionedCall:output:0b_conv3_2_28413b_conv3_2_28415*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????# *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_281672#
!B_conv3_2/StatefulPartitionedCall
B_pool2_2/PartitionedCallPartitionedCall*B_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_281182
B_pool2_2/PartitionedCallπ
B_out/PartitionedCallPartitionedCall"B_pool2_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_B_out_layer_call_and_return_conditional_losses_281902
B_out/PartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCallB_out/PartitionedCall:output:0dense_28420dense_28422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_282092
dense/StatefulPartitionedCallω
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_282422
dropout/PartitionedCall―
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_28426dense_1_28428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_282662!
dense_1/StatefulPartitionedCall·
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_28431dense_2_28433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_282932!
dense_2/StatefulPartitionedCall¨
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^B_conv3_1/StatefulPartitionedCall"^B_conv3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::2F
!B_conv3_1/StatefulPartitionedCall!B_conv3_1/StatefulPartitionedCall2F
!B_conv3_2/StatefulPartitionedCall!B_conv3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????Θ
 
_user_specified_nameinputs
ώ

a
B__inference_dropout_layer_call_and_return_conditional_losses_28237

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΄
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΎ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
α
|
'__inference_dense_1_layer_call_fn_28750

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallϋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_282662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
	
ϊ
,__inference_functional_1_layer_call_fn_28460
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_284372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:?????????Θ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????Θ
!
_user_specified_name	input_1"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
H
input_1=
serving_default_input_1:0?????????Θ;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:―
N
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*q&call_and_return_all_conditional_losses
r_default_save_signature
s__call__"²J
_tf_keras_networkJ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "B_conv3_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv3_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "B_pool2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 3, 3]}, "data_format": "channels_last"}, "name": "B_pool2_1", "inbound_nodes": [[["B_conv3_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "B_conv3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv3_2", "inbound_nodes": [[["B_pool2_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "B_pool2_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2, 2]}, "data_format": "channels_last"}, "name": "B_pool2_2", "inbound_nodes": [[["B_conv3_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "B_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "B_out", "inbound_nodes": [[["B_pool2_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["B_out", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 30, 30, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "B_conv3_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv3_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "B_pool2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 3, 3]}, "data_format": "channels_last"}, "name": "B_pool2_1", "inbound_nodes": [[["B_conv3_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "B_conv3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv3_2", "inbound_nodes": [[["B_pool2_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "B_pool2_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2, 2]}, "data_format": "channels_last"}, "name": "B_pool2_2", "inbound_nodes": [[["B_conv3_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "B_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "B_out", "inbound_nodes": [[["B_pool2_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["B_out", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
"
_tf_keras_input_layerΰ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}



kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"ά
_tf_keras_layerΒ{"class_name": "Conv3D", "name": "B_conv3_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_conv3_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 30, 30, 1]}}
ω
trainable_variables
regularization_losses
	variables
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"κ
_tf_keras_layerΠ{"class_name": "MaxPooling3D", "name": "B_pool2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_pool2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}



kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
*x&call_and_return_all_conditional_losses
y__call__"Ϋ
_tf_keras_layerΑ{"class_name": "Conv3D", "name": "B_conv3_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_conv3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 39, 8, 8, 32]}}
ω
!trainable_variables
"regularization_losses
#	variables
$	keras_api
*z&call_and_return_all_conditional_losses
{__call__"κ
_tf_keras_layerΠ{"class_name": "MaxPooling3D", "name": "B_pool2_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_pool2_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ή
%trainable_variables
&regularization_losses
'	variables
(	keras_api
*|&call_and_return_all_conditional_losses
}__call__"Ο
_tf_keras_layer΅{"class_name": "Flatten", "name": "B_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
π

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
*~&call_and_return_all_conditional_losses
__call__"Λ
_tf_keras_layer±{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1408}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1408]}}
γ
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+&call_and_return_all_conditional_losses
__call__"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ρ

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
+&call_and_return_all_conditional_losses
__call__"Κ
_tf_keras_layer°{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ς

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+&call_and_return_all_conditional_losses
__call__"Λ
_tf_keras_layer±{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
"
	optimizer
f
0
1
2
3
)4
*5
36
47
98
:9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
)4
*5
36
47
98
:9"
trackable_list_wrapper
Κ
trainable_variables

?layers
@non_trainable_variables
regularization_losses
Alayer_metrics
	variables
Blayer_regularization_losses
Cmetrics
s__call__
r_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
.:, 2B_conv3_1/kernel
: 2B_conv3_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Dlayers
trainable_variables
Enon_trainable_variables
Flayer_metrics
regularization_losses
	variables
Glayer_regularization_losses
Hmetrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Ilayers
trainable_variables
Jnon_trainable_variables
Klayer_metrics
regularization_losses
	variables
Llayer_regularization_losses
Mmetrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
.:,  2B_conv3_2/kernel
: 2B_conv3_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Nlayers
trainable_variables
Onon_trainable_variables
Player_metrics
regularization_losses
	variables
Qlayer_regularization_losses
Rmetrics
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Slayers
!trainable_variables
Tnon_trainable_variables
Ulayer_metrics
"regularization_losses
#	variables
Vlayer_regularization_losses
Wmetrics
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Xlayers
%trainable_variables
Ynon_trainable_variables
Zlayer_metrics
&regularization_losses
'	variables
[layer_regularization_losses
\metrics
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
:	@2dense/kernel
:@2
dense/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
­

]layers
+trainable_variables
^non_trainable_variables
_layer_metrics
,regularization_losses
-	variables
`layer_regularization_losses
ametrics
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

blayers
/trainable_variables
cnon_trainable_variables
dlayer_metrics
0regularization_losses
1	variables
elayer_regularization_losses
fmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_1/kernel
:2dense_1/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
°

glayers
5trainable_variables
hnon_trainable_variables
ilayer_metrics
6regularization_losses
7	variables
jlayer_regularization_losses
kmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
°

llayers
;trainable_variables
mnon_trainable_variables
nlayer_metrics
<regularization_losses
=	variables
olayer_regularization_losses
pmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
κ2η
G__inference_functional_1_layer_call_and_return_conditional_losses_28310
G__inference_functional_1_layer_call_and_return_conditional_losses_28343
G__inference_functional_1_layer_call_and_return_conditional_losses_28538
G__inference_functional_1_layer_call_and_return_conditional_losses_28582ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
λ2θ
 __inference__wrapped_model_28100Γ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *3’0
.+
input_1?????????Θ
ώ2ϋ
,__inference_functional_1_layer_call_fn_28402
,__inference_functional_1_layer_call_fn_28632
,__inference_functional_1_layer_call_fn_28607
,__inference_functional_1_layer_call_fn_28460ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_28643’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_B_conv3_1_layer_call_fn_28652’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ή2Ά
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_28106ν
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *M’J
HEA?????????????????????????????????????????????
2
)__inference_B_pool2_1_layer_call_fn_28112ν
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *M’J
HEA?????????????????????????????????????????????
ξ2λ
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_28663’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_B_conv3_2_layer_call_fn_28672’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ή2Ά
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_28118ν
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *M’J
HEA?????????????????????????????????????????????
2
)__inference_B_pool2_2_layer_call_fn_28124ν
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *M’J
HEA?????????????????????????????????????????????
κ2η
@__inference_B_out_layer_call_and_return_conditional_losses_28678’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ο2Μ
%__inference_B_out_layer_call_fn_28683’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_dense_layer_call_and_return_conditional_losses_28694’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ο2Μ
%__inference_dense_layer_call_fn_28703’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Β2Ώ
B__inference_dropout_layer_call_and_return_conditional_losses_28720
B__inference_dropout_layer_call_and_return_conditional_losses_28715΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
'__inference_dropout_layer_call_fn_28725
'__inference_dropout_layer_call_fn_28730΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
μ2ι
B__inference_dense_1_layer_call_and_return_conditional_losses_28741’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_dense_1_layer_call_fn_28750’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_dense_2_layer_call_and_return_conditional_losses_28761’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_dense_2_layer_call_fn_28770’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2B0
#__inference_signature_wrapper_28487input_1Ύ
D__inference_B_conv3_1_layer_call_and_return_conditional_losses_28643v<’9
2’/
-*
inputs?????????Θ
ͺ "2’/
(%
0?????????Δ 
 
)__inference_B_conv3_1_layer_call_fn_28652i<’9
2’/
-*
inputs?????????Θ
ͺ "%"?????????Δ Ό
D__inference_B_conv3_2_layer_call_and_return_conditional_losses_28663t;’8
1’.
,)
inputs?????????' 
ͺ "1’.
'$
0?????????# 
 
)__inference_B_conv3_2_layer_call_fn_28672g;’8
1’.
,)
inputs?????????' 
ͺ "$!?????????# ©
@__inference_B_out_layer_call_and_return_conditional_losses_28678e;’8
1’.
,)
inputs????????? 
ͺ "&’#

0?????????
 
%__inference_B_out_layer_call_fn_28683X;’8
1’.
,)
inputs????????? 
ͺ "?????????
D__inference_B_pool2_1_layer_call_and_return_conditional_losses_28106Έ_’\
U’R
PM
inputsA?????????????????????????????????????????????
ͺ "U’R
KH
0A?????????????????????????????????????????????
 Ω
)__inference_B_pool2_1_layer_call_fn_28112«_’\
U’R
PM
inputsA?????????????????????????????????????????????
ͺ "HEA?????????????????????????????????????????????
D__inference_B_pool2_2_layer_call_and_return_conditional_losses_28118Έ_’\
U’R
PM
inputsA?????????????????????????????????????????????
ͺ "U’R
KH
0A?????????????????????????????????????????????
 Ω
)__inference_B_pool2_2_layer_call_fn_28124«_’\
U’R
PM
inputsA?????????????????????????????????????????????
ͺ "HEA?????????????????????????????????????????????’
 __inference__wrapped_model_28100~
)*349:=’:
3’0
.+
input_1?????????Θ
ͺ "1ͺ.
,
dense_2!
dense_2?????????’
B__inference_dense_1_layer_call_and_return_conditional_losses_28741\34/’,
%’"
 
inputs?????????@
ͺ "%’"

0?????????
 z
'__inference_dense_1_layer_call_fn_28750O34/’,
%’"
 
inputs?????????@
ͺ "?????????’
B__inference_dense_2_layer_call_and_return_conditional_losses_28761\9:/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 z
'__inference_dense_2_layer_call_fn_28770O9:/’,
%’"
 
inputs?????????
ͺ "?????????‘
@__inference_dense_layer_call_and_return_conditional_losses_28694])*0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????@
 y
%__inference_dense_layer_call_fn_28703P)*0’-
&’#
!
inputs?????????
ͺ "?????????@’
B__inference_dropout_layer_call_and_return_conditional_losses_28715\3’0
)’&
 
inputs?????????@
p
ͺ "%’"

0?????????@
 ’
B__inference_dropout_layer_call_and_return_conditional_losses_28720\3’0
)’&
 
inputs?????????@
p 
ͺ "%’"

0?????????@
 z
'__inference_dropout_layer_call_fn_28725O3’0
)’&
 
inputs?????????@
p
ͺ "?????????@z
'__inference_dropout_layer_call_fn_28730O3’0
)’&
 
inputs?????????@
p 
ͺ "?????????@Ε
G__inference_functional_1_layer_call_and_return_conditional_losses_28310z
)*349:E’B
;’8
.+
input_1?????????Θ
p

 
ͺ "%’"

0?????????
 Ε
G__inference_functional_1_layer_call_and_return_conditional_losses_28343z
)*349:E’B
;’8
.+
input_1?????????Θ
p 

 
ͺ "%’"

0?????????
 Δ
G__inference_functional_1_layer_call_and_return_conditional_losses_28538y
)*349:D’A
:’7
-*
inputs?????????Θ
p

 
ͺ "%’"

0?????????
 Δ
G__inference_functional_1_layer_call_and_return_conditional_losses_28582y
)*349:D’A
:’7
-*
inputs?????????Θ
p 

 
ͺ "%’"

0?????????
 
,__inference_functional_1_layer_call_fn_28402m
)*349:E’B
;’8
.+
input_1?????????Θ
p

 
ͺ "?????????
,__inference_functional_1_layer_call_fn_28460m
)*349:E’B
;’8
.+
input_1?????????Θ
p 

 
ͺ "?????????
,__inference_functional_1_layer_call_fn_28607l
)*349:D’A
:’7
-*
inputs?????????Θ
p

 
ͺ "?????????
,__inference_functional_1_layer_call_fn_28632l
)*349:D’A
:’7
-*
inputs?????????Θ
p 

 
ͺ "?????????±
#__inference_signature_wrapper_28487
)*349:H’E
’ 
>ͺ;
9
input_1.+
input_1?????????Θ"1ͺ.
,
dense_2!
dense_2?????????