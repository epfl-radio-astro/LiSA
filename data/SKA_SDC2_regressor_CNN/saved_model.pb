Àö
Ñ£
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
¾
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Ê

A_conv3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameA_conv3_1/kernel

$A_conv3_1/kernel/Read/ReadVariableOpReadVariableOpA_conv3_1/kernel**
_output_shapes
: *
dtype0
t
A_conv3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameA_conv3_1/bias
m
"A_conv3_1/bias/Read/ReadVariableOpReadVariableOpA_conv3_1/bias*
_output_shapes
: *
dtype0

A_conv3_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameA_conv3_2/kernel

$A_conv3_2/kernel/Read/ReadVariableOpReadVariableOpA_conv3_2/kernel**
_output_shapes
:  *
dtype0
t
A_conv3_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameA_conv3_2/bias
m
"A_conv3_2/bias/Read/ReadVariableOpReadVariableOpA_conv3_2/bias*
_output_shapes
: *
dtype0

B_conv5x3x3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameB_conv5x3x3_1/kernel

(B_conv5x3x3_1/kernel/Read/ReadVariableOpReadVariableOpB_conv5x3x3_1/kernel**
_output_shapes
:*
dtype0
|
B_conv5x3x3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameB_conv5x3x3_1/bias
u
&B_conv5x3x3_1/bias/Read/ReadVariableOpReadVariableOpB_conv5x3x3_1/bias*
_output_shapes
:*
dtype0

A_conv3_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameA_conv3_3/kernel

$A_conv3_3/kernel/Read/ReadVariableOpReadVariableOpA_conv3_3/kernel**
_output_shapes
:  *
dtype0
t
A_conv3_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameA_conv3_3/bias
m
"A_conv3_3/bias/Read/ReadVariableOpReadVariableOpA_conv3_3/bias*
_output_shapes
: *
dtype0

B_conv5x3x3_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameB_conv5x3x3_2/kernel

(B_conv5x3x3_2/kernel/Read/ReadVariableOpReadVariableOpB_conv5x3x3_2/kernel**
_output_shapes
: *
dtype0
|
B_conv5x3x3_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameB_conv5x3x3_2/bias
u
&B_conv5x3x3_2/bias/Read/ReadVariableOpReadVariableOpB_conv5x3x3_2/bias*
_output_shapes
: *
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
+*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
+*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/A_conv3_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/A_conv3_1/kernel/m

+Adam/A_conv3_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/A_conv3_1/kernel/m**
_output_shapes
: *
dtype0

Adam/A_conv3_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/A_conv3_1/bias/m
{
)Adam/A_conv3_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/A_conv3_1/bias/m*
_output_shapes
: *
dtype0

Adam/A_conv3_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/A_conv3_2/kernel/m

+Adam/A_conv3_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/A_conv3_2/kernel/m**
_output_shapes
:  *
dtype0

Adam/A_conv3_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/A_conv3_2/bias/m
{
)Adam/A_conv3_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/A_conv3_2/bias/m*
_output_shapes
: *
dtype0

Adam/B_conv5x3x3_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/B_conv5x3x3_1/kernel/m

/Adam/B_conv5x3x3_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_1/kernel/m**
_output_shapes
:*
dtype0

Adam/B_conv5x3x3_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/B_conv5x3x3_1/bias/m

-Adam/B_conv5x3x3_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_1/bias/m*
_output_shapes
:*
dtype0

Adam/A_conv3_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/A_conv3_3/kernel/m

+Adam/A_conv3_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/A_conv3_3/kernel/m**
_output_shapes
:  *
dtype0

Adam/A_conv3_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/A_conv3_3/bias/m
{
)Adam/A_conv3_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/A_conv3_3/bias/m*
_output_shapes
: *
dtype0

Adam/B_conv5x3x3_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/B_conv5x3x3_2/kernel/m

/Adam/B_conv5x3x3_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_2/kernel/m**
_output_shapes
: *
dtype0

Adam/B_conv5x3x3_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/B_conv5x3x3_2/bias/m

-Adam/B_conv5x3x3_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_2/bias/m*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
+*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
+*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/A_conv3_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/A_conv3_1/kernel/v

+Adam/A_conv3_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/A_conv3_1/kernel/v**
_output_shapes
: *
dtype0

Adam/A_conv3_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/A_conv3_1/bias/v
{
)Adam/A_conv3_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/A_conv3_1/bias/v*
_output_shapes
: *
dtype0

Adam/A_conv3_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/A_conv3_2/kernel/v

+Adam/A_conv3_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/A_conv3_2/kernel/v**
_output_shapes
:  *
dtype0

Adam/A_conv3_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/A_conv3_2/bias/v
{
)Adam/A_conv3_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/A_conv3_2/bias/v*
_output_shapes
: *
dtype0

Adam/B_conv5x3x3_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/B_conv5x3x3_1/kernel/v

/Adam/B_conv5x3x3_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_1/kernel/v**
_output_shapes
:*
dtype0

Adam/B_conv5x3x3_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/B_conv5x3x3_1/bias/v

-Adam/B_conv5x3x3_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_1/bias/v*
_output_shapes
:*
dtype0

Adam/A_conv3_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/A_conv3_3/kernel/v

+Adam/A_conv3_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/A_conv3_3/kernel/v**
_output_shapes
:  *
dtype0

Adam/A_conv3_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/A_conv3_3/bias/v
{
)Adam/A_conv3_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/A_conv3_3/bias/v*
_output_shapes
: *
dtype0

Adam/B_conv5x3x3_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/B_conv5x3x3_2/kernel/v

/Adam/B_conv5x3x3_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_2/kernel/v**
_output_shapes
: *
dtype0

Adam/B_conv5x3x3_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/B_conv5x3x3_2/bias/v

-Adam/B_conv5x3x3_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/B_conv5x3x3_2/bias/v*
_output_shapes
: *
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
+*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
+*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
òn
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*­n
value£nB n Bn
Ý
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
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-5
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
h

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
R
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
R
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
R
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
R
^	variables
_regularization_losses
`trainable_variables
a	keras_api
R
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
h

fkernel
gbias
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
R
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
h

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
h

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api

|iter

}beta_1

~beta_2
	decay
learning_ratemõmö&m÷'mø4mù5mú:mû;müHmýImþfmÿgmpmqmvmwmvv&v'v4v5v:v;vHvIvfvgvpvqvvvwv
v
0
1
&2
'3
44
55
:6
;7
H8
I9
f10
g11
p12
q13
v14
w15
 
v
0
1
&2
'3
44
55
:6
;7
H8
I9
f10
g11
p12
q13
v14
w15
²
 layer_regularization_losses
	variables
layer_metrics
regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
 
\Z
VARIABLE_VALUEA_conv3_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEA_conv3_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
²
 layer_regularization_losses
	variables
layer_metrics
regularization_losses
 trainable_variables
metrics
non_trainable_variables
layers
 
 
 
²
 layer_regularization_losses
"	variables
layer_metrics
#regularization_losses
$trainable_variables
metrics
non_trainable_variables
layers
\Z
VARIABLE_VALUEA_conv3_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEA_conv3_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
²
 layer_regularization_losses
(	variables
layer_metrics
)regularization_losses
*trainable_variables
metrics
non_trainable_variables
layers
 
 
 
²
 layer_regularization_losses
,	variables
layer_metrics
-regularization_losses
.trainable_variables
metrics
non_trainable_variables
layers
 
 
 
²
 layer_regularization_losses
0	variables
layer_metrics
1regularization_losses
2trainable_variables
metrics
non_trainable_variables
layers
`^
VARIABLE_VALUEB_conv5x3x3_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEB_conv5x3x3_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
²
 layer_regularization_losses
6	variables
 layer_metrics
7regularization_losses
8trainable_variables
¡metrics
¢non_trainable_variables
£layers
\Z
VARIABLE_VALUEA_conv3_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEA_conv3_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
²
 ¤layer_regularization_losses
<	variables
¥layer_metrics
=regularization_losses
>trainable_variables
¦metrics
§non_trainable_variables
¨layers
 
 
 
²
 ©layer_regularization_losses
@	variables
ªlayer_metrics
Aregularization_losses
Btrainable_variables
«metrics
¬non_trainable_variables
­layers
 
 
 
²
 ®layer_regularization_losses
D	variables
¯layer_metrics
Eregularization_losses
Ftrainable_variables
°metrics
±non_trainable_variables
²layers
`^
VARIABLE_VALUEB_conv5x3x3_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEB_conv5x3x3_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
 

H0
I1
²
 ³layer_regularization_losses
J	variables
´layer_metrics
Kregularization_losses
Ltrainable_variables
µmetrics
¶non_trainable_variables
·layers
 
 
 
²
 ¸layer_regularization_losses
N	variables
¹layer_metrics
Oregularization_losses
Ptrainable_variables
ºmetrics
»non_trainable_variables
¼layers
 
 
 
²
 ½layer_regularization_losses
R	variables
¾layer_metrics
Sregularization_losses
Ttrainable_variables
¿metrics
Ànon_trainable_variables
Álayers
 
 
 
²
 Âlayer_regularization_losses
V	variables
Ãlayer_metrics
Wregularization_losses
Xtrainable_variables
Ämetrics
Ånon_trainable_variables
Ælayers
 
 
 
²
 Çlayer_regularization_losses
Z	variables
Èlayer_metrics
[regularization_losses
\trainable_variables
Émetrics
Ênon_trainable_variables
Ëlayers
 
 
 
²
 Ìlayer_regularization_losses
^	variables
Ílayer_metrics
_regularization_losses
`trainable_variables
Îmetrics
Ïnon_trainable_variables
Ðlayers
 
 
 
²
 Ñlayer_regularization_losses
b	variables
Òlayer_metrics
cregularization_losses
dtrainable_variables
Ómetrics
Ônon_trainable_variables
Õlayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
²
 Ölayer_regularization_losses
h	variables
×layer_metrics
iregularization_losses
jtrainable_variables
Ømetrics
Ùnon_trainable_variables
Úlayers
 
 
 
²
 Ûlayer_regularization_losses
l	variables
Ülayer_metrics
mregularization_losses
ntrainable_variables
Ýmetrics
Þnon_trainable_variables
ßlayers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
²
 àlayer_regularization_losses
r	variables
álayer_metrics
sregularization_losses
ttrainable_variables
âmetrics
ãnon_trainable_variables
älayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
 

v0
w1
²
 ålayer_regularization_losses
x	variables
ælayer_metrics
yregularization_losses
ztrainable_variables
çmetrics
ènon_trainable_variables
élayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

ê0
ë1
 

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
10
11
12
13
14
15
16
17
18
19
20
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
 
 
8

ìtotal

ícount
î	variables
ï	keras_api
I

ðtotal

ñcount
ò
_fn_kwargs
ó	variables
ô	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ì0
í1

î	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ð0
ñ1

ó	variables
}
VARIABLE_VALUEAdam/A_conv3_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/A_conv3_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/A_conv3_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/A_conv3_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/B_conv5x3x3_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/B_conv5x3x3_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/A_conv3_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/A_conv3_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/B_conv5x3x3_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/B_conv5x3x3_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/A_conv3_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/A_conv3_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/A_conv3_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/A_conv3_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/B_conv5x3x3_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/B_conv5x3x3_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/A_conv3_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/A_conv3_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/B_conv5x3x3_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/B_conv5x3x3_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ*
dtype0*)
shape :ÿÿÿÿÿÿÿÿÿÈ
ë
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1A_conv3_1/kernelA_conv3_1/biasA_conv3_2/kernelA_conv3_2/biasB_conv5x3x3_1/kernelB_conv5x3x3_1/biasA_conv3_3/kernelA_conv3_3/biasB_conv5x3x3_2/kernelB_conv5x3x3_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *-
f(R&
$__inference_signature_wrapper_631735
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$A_conv3_1/kernel/Read/ReadVariableOp"A_conv3_1/bias/Read/ReadVariableOp$A_conv3_2/kernel/Read/ReadVariableOp"A_conv3_2/bias/Read/ReadVariableOp(B_conv5x3x3_1/kernel/Read/ReadVariableOp&B_conv5x3x3_1/bias/Read/ReadVariableOp$A_conv3_3/kernel/Read/ReadVariableOp"A_conv3_3/bias/Read/ReadVariableOp(B_conv5x3x3_2/kernel/Read/ReadVariableOp&B_conv5x3x3_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/A_conv3_1/kernel/m/Read/ReadVariableOp)Adam/A_conv3_1/bias/m/Read/ReadVariableOp+Adam/A_conv3_2/kernel/m/Read/ReadVariableOp)Adam/A_conv3_2/bias/m/Read/ReadVariableOp/Adam/B_conv5x3x3_1/kernel/m/Read/ReadVariableOp-Adam/B_conv5x3x3_1/bias/m/Read/ReadVariableOp+Adam/A_conv3_3/kernel/m/Read/ReadVariableOp)Adam/A_conv3_3/bias/m/Read/ReadVariableOp/Adam/B_conv5x3x3_2/kernel/m/Read/ReadVariableOp-Adam/B_conv5x3x3_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp+Adam/A_conv3_1/kernel/v/Read/ReadVariableOp)Adam/A_conv3_1/bias/v/Read/ReadVariableOp+Adam/A_conv3_2/kernel/v/Read/ReadVariableOp)Adam/A_conv3_2/bias/v/Read/ReadVariableOp/Adam/B_conv5x3x3_1/kernel/v/Read/ReadVariableOp-Adam/B_conv5x3x3_1/bias/v/Read/ReadVariableOp+Adam/A_conv3_3/kernel/v/Read/ReadVariableOp)Adam/A_conv3_3/bias/v/Read/ReadVariableOp/Adam/B_conv5x3x3_2/kernel/v/Read/ReadVariableOp-Adam/B_conv5x3x3_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
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
GPU2*0,1,2,3J 8 *(
f#R!
__inference__traced_save_632607
Í
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameA_conv3_1/kernelA_conv3_1/biasA_conv3_2/kernelA_conv3_2/biasB_conv5x3x3_1/kernelB_conv5x3x3_1/biasA_conv3_3/kernelA_conv3_3/biasB_conv5x3x3_2/kernelB_conv5x3x3_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/A_conv3_1/kernel/mAdam/A_conv3_1/bias/mAdam/A_conv3_2/kernel/mAdam/A_conv3_2/bias/mAdam/B_conv5x3x3_1/kernel/mAdam/B_conv5x3x3_1/bias/mAdam/A_conv3_3/kernel/mAdam/A_conv3_3/bias/mAdam/B_conv5x3x3_2/kernel/mAdam/B_conv5x3x3_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/A_conv3_1/kernel/vAdam/A_conv3_1/bias/vAdam/A_conv3_2/kernel/vAdam/A_conv3_2/bias/vAdam/B_conv5x3x3_1/kernel/vAdam/B_conv5x3x3_1/bias/vAdam/A_conv3_3/kernel/vAdam/A_conv3_3/bias/vAdam/B_conv5x3x3_2/kernel/vAdam/B_conv5x3x3_2/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*E
Tin>
<2:*
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
GPU2*0,1,2,3J 8 *+
f&R$
"__inference__traced_restore_632788ü

D
(__inference_dropout_layer_call_fn_632327

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6313172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs
ý
m
4__inference_spatial_dropout3d_1_layer_call_fn_632260

inputs
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6310192
StatefulPartitionedCall¾
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
üQ
Ï
H__inference_functional_3_layer_call_and_return_conditional_losses_631935

inputs,
(a_conv3_1_conv3d_readvariableop_resource-
)a_conv3_1_biasadd_readvariableop_resource,
(a_conv3_2_conv3d_readvariableop_resource-
)a_conv3_2_biasadd_readvariableop_resource0
,b_conv5x3x3_1_conv3d_readvariableop_resource1
-b_conv5x3x3_1_biasadd_readvariableop_resource,
(a_conv3_3_conv3d_readvariableop_resource-
)a_conv3_3_biasadd_readvariableop_resource0
,b_conv5x3x3_2_conv3d_readvariableop_resource1
-b_conv5x3x3_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity·
A_conv3_1/Conv3D/ReadVariableOpReadVariableOp(a_conv3_1_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02!
A_conv3_1/Conv3D/ReadVariableOpÈ
A_conv3_1/Conv3DConv3Dinputs'A_conv3_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *
paddingVALID*
strides	
2
A_conv3_1/Conv3Dª
 A_conv3_1/BiasAdd/ReadVariableOpReadVariableOp)a_conv3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 A_conv3_1/BiasAdd/ReadVariableOpµ
A_conv3_1/BiasAddBiasAddA_conv3_1/Conv3D:output:0(A_conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2
A_conv3_1/BiasAdd
A_conv3_1/ReluReluA_conv3_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2
A_conv3_1/ReluÑ
A_pool2_1/MaxPool3D	MaxPool3DA_conv3_1/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc *
ksize	
*
paddingVALID*
strides	
2
A_pool2_1/MaxPool3D»
B_celpool/AvgPool3D	AvgPool3Dinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(*
ksize	
*
paddingVALID*
strides	
2
B_celpool/AvgPool3D·
A_conv3_2/Conv3D/ReadVariableOpReadVariableOp(a_conv3_2_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
A_conv3_2/Conv3D/ReadVariableOpÝ
A_conv3_2/Conv3DConv3DA_pool2_1/MaxPool3D:output:0'A_conv3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *
paddingVALID*
strides	
2
A_conv3_2/Conv3Dª
 A_conv3_2/BiasAdd/ReadVariableOpReadVariableOp)a_conv3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 A_conv3_2/BiasAdd/ReadVariableOp´
A_conv3_2/BiasAddBiasAddA_conv3_2/Conv3D:output:0(A_conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2
A_conv3_2/BiasAdd
A_conv3_2/ReluReluA_conv3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2
A_conv3_2/ReluÃ
#B_conv5x3x3_1/Conv3D/ReadVariableOpReadVariableOp,b_conv5x3x3_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02%
#B_conv5x3x3_1/Conv3D/ReadVariableOpé
B_conv5x3x3_1/Conv3DConv3DB_celpool/AvgPool3D:output:0+B_conv5x3x3_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*
paddingVALID*
strides	
2
B_conv5x3x3_1/Conv3D¶
$B_conv5x3x3_1/BiasAdd/ReadVariableOpReadVariableOp-b_conv5x3x3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$B_conv5x3x3_1/BiasAdd/ReadVariableOpÄ
B_conv5x3x3_1/BiasAddBiasAddB_conv5x3x3_1/Conv3D:output:0,B_conv5x3x3_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2
B_conv5x3x3_1/BiasAdd
B_conv5x3x3_1/ReluReluB_conv5x3x3_1/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2
B_conv5x3x3_1/ReluÑ
A_pool2_2/MaxPool3D	MaxPool3DA_conv3_2/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 *
ksize	
*
paddingVALID*
strides	
2
A_pool2_2/MaxPool3DÝ
B_pool4x2x2_1/MaxPool3D	MaxPool3D B_conv5x3x3_1/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2
B_pool4x2x2_1/MaxPool3D·
A_conv3_3/Conv3D/ReadVariableOpReadVariableOp(a_conv3_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
A_conv3_3/Conv3D/ReadVariableOpÝ
A_conv3_3/Conv3DConv3DA_pool2_2/MaxPool3D:output:0'A_conv3_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *
paddingVALID*
strides	
2
A_conv3_3/Conv3Dª
 A_conv3_3/BiasAdd/ReadVariableOpReadVariableOp)a_conv3_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 A_conv3_3/BiasAdd/ReadVariableOp´
A_conv3_3/BiasAddBiasAddA_conv3_3/Conv3D:output:0(A_conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
A_conv3_3/BiasAdd
A_conv3_3/ReluReluA_conv3_3/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
A_conv3_3/ReluÃ
#B_conv5x3x3_2/Conv3D/ReadVariableOpReadVariableOp,b_conv5x3x3_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02%
#B_conv5x3x3_2/Conv3D/ReadVariableOpí
B_conv5x3x3_2/Conv3DConv3D B_pool4x2x2_1/MaxPool3D:output:0+B_conv5x3x3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides	
2
B_conv5x3x3_2/Conv3D¶
$B_conv5x3x3_2/BiasAdd/ReadVariableOpReadVariableOp-b_conv5x3x3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$B_conv5x3x3_2/BiasAdd/ReadVariableOpÄ
B_conv5x3x3_2/BiasAddBiasAddB_conv5x3x3_2/Conv3D:output:0,B_conv5x3x3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
B_conv5x3x3_2/BiasAdd
B_conv5x3x3_2/ReluReluB_conv5x3x3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
B_conv5x3x3_2/Relu 
spatial_dropout3d/IdentityIdentityA_conv3_3/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
spatial_dropout3d/Identity¨
spatial_dropout3d_1/IdentityIdentity B_conv5x3x3_2/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
spatial_dropout3d_1/IdentityØ
A_pool2_3/MaxPool3D	MaxPool3D#spatial_dropout3d/Identity:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
ksize	
*
paddingVALID*
strides	
2
A_pool2_3/MaxPool3Dk
A_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
A_out/Const
A_out/ReshapeReshapeA_pool2_3/MaxPool3D:output:0A_out/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
A_out/Reshapek
B_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
B_out/Const
B_out/ReshapeReshape%spatial_dropout3d_1/Identity:output:0B_out/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
B_out/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÂ
concatenate/concatConcatV2A_out/Reshape:output:0B_out/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
concatenate/concat
dropout/IdentityIdentityconcatenate/concat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/Identity¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
+*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu
dropout_1/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Identity§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp¡
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ:::::::::::::::::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ê
l
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632152

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ. :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 
 
_user_specified_nameinputs
ì
n
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_631238

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

k
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_631203

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity_1"!

identity_1Identity_1:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ. :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 
 
_user_specified_nameinputs
­
c
*__inference_dropout_1_layer_call_fn_632369

inputs
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6313692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

m
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_631243

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â
a
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_630838

inputs
identityË
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

.__inference_B_conv5x3x3_1_layer_call_fn_632069

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_6311032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ(::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
 
l
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_630937

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const£
dropout/MulMulinputsdropout/Const:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Castª
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
©
A__inference_dense_layer_call_and_return_conditional_losses_632338

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
+*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs


*__inference_A_conv3_2_layer_call_fn_632049

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_6310762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿc ::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc 
 
_user_specified_nameinputs
í
m
4__inference_spatial_dropout3d_1_layer_call_fn_632221

inputs
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6312382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®	
­
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_632020

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOpª
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *
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
 :ÿÿÿÿÿÿÿÿÿÆ 2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¥

Ñ
$__inference_signature_wrapper_631735
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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 **
f%R#
!__inference__wrapped_model_6308322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
®X

H__inference_functional_3_layer_call_and_return_conditional_losses_631441
input_1
a_conv3_1_631058
a_conv3_1_631060
a_conv3_2_631087
a_conv3_2_631089
b_conv5x3x3_1_631114
b_conv5x3x3_1_631116
a_conv3_3_631143
a_conv3_3_631145
b_conv5x3x3_2_631170
b_conv5x3x3_2_631172
dense_631352
dense_631354
dense_1_631409
dense_1_631411
dense_2_631435
dense_2_631437
identity¢!A_conv3_1/StatefulPartitionedCall¢!A_conv3_2/StatefulPartitionedCall¢!A_conv3_3/StatefulPartitionedCall¢%B_conv5x3x3_1/StatefulPartitionedCall¢%B_conv5x3x3_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢)spatial_dropout3d/StatefulPartitionedCall¢+spatial_dropout3d_1/StatefulPartitionedCall°
!A_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinput_1a_conv3_1_631058a_conv3_1_631060*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_6310472#
!A_conv3_1/StatefulPartitionedCall
A_pool2_1/PartitionedCallPartitionedCall*A_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_6308382
A_pool2_1/PartitionedCallí
B_celpool/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_B_celpool_layer_call_and_return_conditional_losses_6308502
B_celpool/PartitionedCallÊ
!A_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_1/PartitionedCall:output:0a_conv3_2_631087a_conv3_2_631089*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_6310762#
!A_conv3_2/StatefulPartitionedCallÞ
%B_conv5x3x3_1/StatefulPartitionedCallStatefulPartitionedCall"B_celpool/PartitionedCall:output:0b_conv5x3x3_1_631114b_conv5x3x3_1_631116*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_6311032'
%B_conv5x3x3_1/StatefulPartitionedCall
A_pool2_2/PartitionedCallPartitionedCall*A_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_6308622
A_pool2_2/PartitionedCall 
B_pool4x2x2_1/PartitionedCallPartitionedCall.B_conv5x3x3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_6308742
B_pool4x2x2_1/PartitionedCallÊ
!A_conv3_3/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_2/PartitionedCall:output:0a_conv3_3_631143a_conv3_3_631145*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_6311322#
!A_conv3_3/StatefulPartitionedCallâ
%B_conv5x3x3_2/StatefulPartitionedCallStatefulPartitionedCall&B_pool4x2x2_1/PartitionedCall:output:0b_conv5x3x3_2_631170b_conv5x3x3_2_631172*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_6311592'
%B_conv5x3x3_2/StatefulPartitionedCallÀ
)spatial_dropout3d/StatefulPartitionedCallStatefulPartitionedCall*A_conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6311982+
)spatial_dropout3d/StatefulPartitionedCallö
+spatial_dropout3d_1/StatefulPartitionedCallStatefulPartitionedCall.B_conv5x3x3_2/StatefulPartitionedCall:output:0*^spatial_dropout3d/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6312382-
+spatial_dropout3d_1/StatefulPartitionedCall
A_pool2_3/PartitionedCallPartitionedCall2spatial_dropout3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_6309562
A_pool2_3/PartitionedCallñ
A_out/PartitionedCallPartitionedCall"A_pool2_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_A_out_layer_call_and_return_conditional_losses_6312622
A_out/PartitionedCall
B_out/PartitionedCallPartitionedCall4spatial_dropout3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_B_out_layer_call_and_return_conditional_losses_6312762
B_out/PartitionedCall 
concatenate/PartitionedCallPartitionedCallA_out/PartitionedCall:output:0B_out/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6312912
concatenate/PartitionedCall¿
dropout/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0,^spatial_dropout3d_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6313122!
dropout/StatefulPartitionedCall±
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_631352dense_631354*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6313412
dense/StatefulPartitionedCall»
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6313692#
!dropout_1/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_631409dense_1_631411*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6313982!
dense_1/StatefulPartitionedCallº
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_631435dense_2_631437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6314242!
dense_2/StatefulPartitionedCall¼
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^A_conv3_1/StatefulPartitionedCall"^A_conv3_2/StatefulPartitionedCall"^A_conv3_3/StatefulPartitionedCall&^B_conv5x3x3_1/StatefulPartitionedCall&^B_conv5x3x3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^spatial_dropout3d/StatefulPartitionedCall,^spatial_dropout3d_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::2F
!A_conv3_1/StatefulPartitionedCall!A_conv3_1/StatefulPartitionedCall2F
!A_conv3_2/StatefulPartitionedCall!A_conv3_2/StatefulPartitionedCall2F
!A_conv3_3/StatefulPartitionedCall!A_conv3_3/StatefulPartitionedCall2N
%B_conv5x3x3_1/StatefulPartitionedCall%B_conv5x3x3_1/StatefulPartitionedCall2N
%B_conv5x3x3_2/StatefulPartitionedCall%B_conv5x3x3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)spatial_dropout3d/StatefulPartitionedCall)spatial_dropout3d/StatefulPartitionedCall2Z
+spatial_dropout3d_1/StatefulPartitionedCall+spatial_dropout3d_1/StatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1

m
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_631029

inputs

identity_1
IdentityIdentityinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

k
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_630947

inputs

identity_1
IdentityIdentityinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

m
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632216

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±
«
C__inference_dense_1_layer_call_and_return_conditional_losses_632385

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_A_conv3_1_layer_call_fn_632029

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_6310472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_631369

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
F
*__inference_B_celpool_layer_call_fn_630856

inputs
identityü
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_B_celpool_layer_call_and_return_conditional_losses_6308502
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
F
*__inference_A_pool2_2_layer_call_fn_630868

inputs
identityü
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_6308622
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
«
C__inference_dense_1_layer_call_and_return_conditional_losses_631398

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
}
(__inference_dense_1_layer_call_fn_632394

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6313982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
«
C__inference_dense_2_layer_call_and_return_conditional_losses_632404

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨	
­
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_631132

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *
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
:ÿÿÿÿÿÿÿÿÿ. 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ0 :::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 
 
_user_specified_nameinputs
ã
{
&__inference_dense_layer_call_fn_632347

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6313412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs
Õ

Ú
-__inference_functional_3_layer_call_fn_631684
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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_6316492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
Ò

Ù
-__inference_functional_3_layer_call_fn_631972

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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_6315562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
 

.__inference_B_conv5x3x3_2_layer_call_fn_632187

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_6311592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
n
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_631019

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const£
dropout/MulMulinputsdropout/Const:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Castª
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
]
A__inference_B_out_layer_call_and_return_conditional_losses_631276

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ì
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_631374

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
a
(__inference_dropout_layer_call_fn_632322

inputs
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6313122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs
ªX

H__inference_functional_3_layer_call_and_return_conditional_losses_631556

inputs
a_conv3_1_631503
a_conv3_1_631505
a_conv3_2_631510
a_conv3_2_631512
b_conv5x3x3_1_631515
b_conv5x3x3_1_631517
a_conv3_3_631522
a_conv3_3_631524
b_conv5x3x3_2_631527
b_conv5x3x3_2_631529
dense_631539
dense_631541
dense_1_631545
dense_1_631547
dense_2_631550
dense_2_631552
identity¢!A_conv3_1/StatefulPartitionedCall¢!A_conv3_2/StatefulPartitionedCall¢!A_conv3_3/StatefulPartitionedCall¢%B_conv5x3x3_1/StatefulPartitionedCall¢%B_conv5x3x3_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢)spatial_dropout3d/StatefulPartitionedCall¢+spatial_dropout3d_1/StatefulPartitionedCall¯
!A_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinputsa_conv3_1_631503a_conv3_1_631505*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_6310472#
!A_conv3_1/StatefulPartitionedCall
A_pool2_1/PartitionedCallPartitionedCall*A_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_6308382
A_pool2_1/PartitionedCallì
B_celpool/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_B_celpool_layer_call_and_return_conditional_losses_6308502
B_celpool/PartitionedCallÊ
!A_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_1/PartitionedCall:output:0a_conv3_2_631510a_conv3_2_631512*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_6310762#
!A_conv3_2/StatefulPartitionedCallÞ
%B_conv5x3x3_1/StatefulPartitionedCallStatefulPartitionedCall"B_celpool/PartitionedCall:output:0b_conv5x3x3_1_631515b_conv5x3x3_1_631517*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_6311032'
%B_conv5x3x3_1/StatefulPartitionedCall
A_pool2_2/PartitionedCallPartitionedCall*A_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_6308622
A_pool2_2/PartitionedCall 
B_pool4x2x2_1/PartitionedCallPartitionedCall.B_conv5x3x3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_6308742
B_pool4x2x2_1/PartitionedCallÊ
!A_conv3_3/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_2/PartitionedCall:output:0a_conv3_3_631522a_conv3_3_631524*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_6311322#
!A_conv3_3/StatefulPartitionedCallâ
%B_conv5x3x3_2/StatefulPartitionedCallStatefulPartitionedCall&B_pool4x2x2_1/PartitionedCall:output:0b_conv5x3x3_2_631527b_conv5x3x3_2_631529*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_6311592'
%B_conv5x3x3_2/StatefulPartitionedCallÀ
)spatial_dropout3d/StatefulPartitionedCallStatefulPartitionedCall*A_conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6311982+
)spatial_dropout3d/StatefulPartitionedCallö
+spatial_dropout3d_1/StatefulPartitionedCallStatefulPartitionedCall.B_conv5x3x3_2/StatefulPartitionedCall:output:0*^spatial_dropout3d/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6312382-
+spatial_dropout3d_1/StatefulPartitionedCall
A_pool2_3/PartitionedCallPartitionedCall2spatial_dropout3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_6309562
A_pool2_3/PartitionedCallñ
A_out/PartitionedCallPartitionedCall"A_pool2_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_A_out_layer_call_and_return_conditional_losses_6312622
A_out/PartitionedCall
B_out/PartitionedCallPartitionedCall4spatial_dropout3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_B_out_layer_call_and_return_conditional_losses_6312762
B_out/PartitionedCall 
concatenate/PartitionedCallPartitionedCallA_out/PartitionedCall:output:0B_out/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6312912
concatenate/PartitionedCall¿
dropout/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0,^spatial_dropout3d_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6313122!
dropout/StatefulPartitionedCall±
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_631539dense_631541*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6313412
dense/StatefulPartitionedCall»
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6313692#
!dropout_1/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_631545dense_1_631547*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6313982!
dense_1/StatefulPartitionedCallº
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_631550dense_2_631552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6314242!
dense_2/StatefulPartitionedCall¼
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^A_conv3_1/StatefulPartitionedCall"^A_conv3_2/StatefulPartitionedCall"^A_conv3_3/StatefulPartitionedCall&^B_conv5x3x3_1/StatefulPartitionedCall&^B_conv5x3x3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^spatial_dropout3d/StatefulPartitionedCall,^spatial_dropout3d_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::2F
!A_conv3_1/StatefulPartitionedCall!A_conv3_1/StatefulPartitionedCall2F
!A_conv3_2/StatefulPartitionedCall!A_conv3_2/StatefulPartitionedCall2F
!A_conv3_3/StatefulPartitionedCall!A_conv3_3/StatefulPartitionedCall2N
%B_conv5x3x3_1/StatefulPartitionedCall%B_conv5x3x3_1/StatefulPartitionedCall2N
%B_conv5x3x3_2/StatefulPartitionedCall%B_conv5x3x3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)spatial_dropout3d/StatefulPartitionedCall)spatial_dropout3d/StatefulPartitionedCall2Z
+spatial_dropout3d_1/StatefulPartitionedCall+spatial_dropout3d_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¢
n
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632250

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const£
dropout/MulMulinputsdropout/Const:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Castª
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
a
C__inference_dropout_layer_call_and_return_conditional_losses_632317

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs


*__inference_A_conv3_3_layer_call_fn_632089

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_6311322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ0 ::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 
 
_user_specified_nameinputs
Þ
F
*__inference_A_pool2_3_layer_call_fn_630962

inputs
identityü
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_6309562
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
P
4__inference_spatial_dropout3d_1_layer_call_fn_632265

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6310292
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
l
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_631198

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ. :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 
 
_user_specified_nameinputs
¡
F
*__inference_dropout_1_layer_call_fn_632374

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6313742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
X
,__inference_concatenate_layer_call_fn_632300
inputs_0
inputs_1
identityÜ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6312912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
å
}
(__inference_dense_2_layer_call_fn_632413

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6314242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨	
­
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_632080

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *
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
:ÿÿÿÿÿÿÿÿÿ. 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ0 :::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 
 
_user_specified_nameinputs
ýt
È
__inference__traced_save_632607
file_prefix/
+savev2_a_conv3_1_kernel_read_readvariableop-
)savev2_a_conv3_1_bias_read_readvariableop/
+savev2_a_conv3_2_kernel_read_readvariableop-
)savev2_a_conv3_2_bias_read_readvariableop3
/savev2_b_conv5x3x3_1_kernel_read_readvariableop1
-savev2_b_conv5x3x3_1_bias_read_readvariableop/
+savev2_a_conv3_3_kernel_read_readvariableop-
)savev2_a_conv3_3_bias_read_readvariableop3
/savev2_b_conv5x3x3_2_kernel_read_readvariableop1
-savev2_b_conv5x3x3_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_a_conv3_1_kernel_m_read_readvariableop4
0savev2_adam_a_conv3_1_bias_m_read_readvariableop6
2savev2_adam_a_conv3_2_kernel_m_read_readvariableop4
0savev2_adam_a_conv3_2_bias_m_read_readvariableop:
6savev2_adam_b_conv5x3x3_1_kernel_m_read_readvariableop8
4savev2_adam_b_conv5x3x3_1_bias_m_read_readvariableop6
2savev2_adam_a_conv3_3_kernel_m_read_readvariableop4
0savev2_adam_a_conv3_3_bias_m_read_readvariableop:
6savev2_adam_b_conv5x3x3_2_kernel_m_read_readvariableop8
4savev2_adam_b_conv5x3x3_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop6
2savev2_adam_a_conv3_1_kernel_v_read_readvariableop4
0savev2_adam_a_conv3_1_bias_v_read_readvariableop6
2savev2_adam_a_conv3_2_kernel_v_read_readvariableop4
0savev2_adam_a_conv3_2_bias_v_read_readvariableop:
6savev2_adam_b_conv5x3x3_1_kernel_v_read_readvariableop8
4savev2_adam_b_conv5x3x3_1_bias_v_read_readvariableop6
2savev2_adam_a_conv3_3_kernel_v_read_readvariableop4
0savev2_adam_a_conv3_3_bias_v_read_readvariableop:
6savev2_adam_b_conv5x3x3_2_kernel_v_read_readvariableop8
4savev2_adam_b_conv5x3x3_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
value3B1 B+_temp_580d9696098749f79a7decb399bdf85a/part2	
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
ShardedFilename® 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*À
value¶B³:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesý
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesë
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_a_conv3_1_kernel_read_readvariableop)savev2_a_conv3_1_bias_read_readvariableop+savev2_a_conv3_2_kernel_read_readvariableop)savev2_a_conv3_2_bias_read_readvariableop/savev2_b_conv5x3x3_1_kernel_read_readvariableop-savev2_b_conv5x3x3_1_bias_read_readvariableop+savev2_a_conv3_3_kernel_read_readvariableop)savev2_a_conv3_3_bias_read_readvariableop/savev2_b_conv5x3x3_2_kernel_read_readvariableop-savev2_b_conv5x3x3_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_a_conv3_1_kernel_m_read_readvariableop0savev2_adam_a_conv3_1_bias_m_read_readvariableop2savev2_adam_a_conv3_2_kernel_m_read_readvariableop0savev2_adam_a_conv3_2_bias_m_read_readvariableop6savev2_adam_b_conv5x3x3_1_kernel_m_read_readvariableop4savev2_adam_b_conv5x3x3_1_bias_m_read_readvariableop2savev2_adam_a_conv3_3_kernel_m_read_readvariableop0savev2_adam_a_conv3_3_bias_m_read_readvariableop6savev2_adam_b_conv5x3x3_2_kernel_m_read_readvariableop4savev2_adam_b_conv5x3x3_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop2savev2_adam_a_conv3_1_kernel_v_read_readvariableop0savev2_adam_a_conv3_1_bias_v_read_readvariableop2savev2_adam_a_conv3_2_kernel_v_read_readvariableop0savev2_adam_a_conv3_2_bias_v_read_readvariableop6savev2_adam_b_conv5x3x3_1_kernel_v_read_readvariableop4savev2_adam_b_conv5x3x3_1_bias_v_read_readvariableop2savev2_adam_a_conv3_3_kernel_v_read_readvariableop0savev2_adam_a_conv3_3_bias_v_read_readvariableop6savev2_adam_b_conv5x3x3_2_kernel_v_read_readvariableop4savev2_adam_b_conv5x3x3_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*ô
_input_shapesâ
ß: : : :  : :::  : : : :
+::
::	:: : : : : : : : : : : :  : :::  : : : :
+::
::	:: : :  : :::  : : : :
+::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:  : 

_output_shapes
: :0	,
*
_output_shapes
: : 


_output_shapes
: :&"
 
_output_shapes
:
+:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
:: 

_output_shapes
::0 ,
*
_output_shapes
:  : !

_output_shapes
: :0",
*
_output_shapes
: : #

_output_shapes
: :&$"
 
_output_shapes
:
+:!%

_output_shapes	
::&&"
 
_output_shapes
:
:!'

_output_shapes	
::%(!

_output_shapes
:	: )

_output_shapes
::0*,
*
_output_shapes
: : +

_output_shapes
: :0,,
*
_output_shapes
:  : -

_output_shapes
: :0.,
*
_output_shapes
:: /

_output_shapes
::00,
*
_output_shapes
:  : 1

_output_shapes
: :02,
*
_output_shapes
: : 3

_output_shapes
: :&4"
 
_output_shapes
:
+:!5

_output_shapes	
::&6"
 
_output_shapes
:
:!7

_output_shapes	
::%8!

_output_shapes
:	: 9

_output_shapes
:::

_output_shapes
: 
î
N
2__inference_spatial_dropout3d_layer_call_fn_632128

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6309472
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
B
&__inference_B_out_layer_call_fn_632287

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_B_out_layer_call_and_return_conditional_losses_6312762
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
é
k
2__inference_spatial_dropout3d_layer_call_fn_632162

inputs
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6311982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ. 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 
 
_user_specified_nameinputs
 
l
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632113

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const£
dropout/MulMulinputsdropout/Const:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Castª
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
©
A__inference_dense_layer_call_and_return_conditional_losses_631341

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
+*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs
Á
]
A__inference_B_out_layer_call_and_return_conditional_losses_632282

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ê£
Ï
H__inference_functional_3_layer_call_and_return_conditional_losses_631861

inputs,
(a_conv3_1_conv3d_readvariableop_resource-
)a_conv3_1_biasadd_readvariableop_resource,
(a_conv3_2_conv3d_readvariableop_resource-
)a_conv3_2_biasadd_readvariableop_resource0
,b_conv5x3x3_1_conv3d_readvariableop_resource1
-b_conv5x3x3_1_biasadd_readvariableop_resource,
(a_conv3_3_conv3d_readvariableop_resource-
)a_conv3_3_biasadd_readvariableop_resource0
,b_conv5x3x3_2_conv3d_readvariableop_resource1
-b_conv5x3x3_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity·
A_conv3_1/Conv3D/ReadVariableOpReadVariableOp(a_conv3_1_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02!
A_conv3_1/Conv3D/ReadVariableOpÈ
A_conv3_1/Conv3DConv3Dinputs'A_conv3_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *
paddingVALID*
strides	
2
A_conv3_1/Conv3Dª
 A_conv3_1/BiasAdd/ReadVariableOpReadVariableOp)a_conv3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 A_conv3_1/BiasAdd/ReadVariableOpµ
A_conv3_1/BiasAddBiasAddA_conv3_1/Conv3D:output:0(A_conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2
A_conv3_1/BiasAdd
A_conv3_1/ReluReluA_conv3_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2
A_conv3_1/ReluÑ
A_pool2_1/MaxPool3D	MaxPool3DA_conv3_1/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc *
ksize	
*
paddingVALID*
strides	
2
A_pool2_1/MaxPool3D»
B_celpool/AvgPool3D	AvgPool3Dinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(*
ksize	
*
paddingVALID*
strides	
2
B_celpool/AvgPool3D·
A_conv3_2/Conv3D/ReadVariableOpReadVariableOp(a_conv3_2_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
A_conv3_2/Conv3D/ReadVariableOpÝ
A_conv3_2/Conv3DConv3DA_pool2_1/MaxPool3D:output:0'A_conv3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *
paddingVALID*
strides	
2
A_conv3_2/Conv3Dª
 A_conv3_2/BiasAdd/ReadVariableOpReadVariableOp)a_conv3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 A_conv3_2/BiasAdd/ReadVariableOp´
A_conv3_2/BiasAddBiasAddA_conv3_2/Conv3D:output:0(A_conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2
A_conv3_2/BiasAdd
A_conv3_2/ReluReluA_conv3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2
A_conv3_2/ReluÃ
#B_conv5x3x3_1/Conv3D/ReadVariableOpReadVariableOp,b_conv5x3x3_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02%
#B_conv5x3x3_1/Conv3D/ReadVariableOpé
B_conv5x3x3_1/Conv3DConv3DB_celpool/AvgPool3D:output:0+B_conv5x3x3_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*
paddingVALID*
strides	
2
B_conv5x3x3_1/Conv3D¶
$B_conv5x3x3_1/BiasAdd/ReadVariableOpReadVariableOp-b_conv5x3x3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$B_conv5x3x3_1/BiasAdd/ReadVariableOpÄ
B_conv5x3x3_1/BiasAddBiasAddB_conv5x3x3_1/Conv3D:output:0,B_conv5x3x3_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2
B_conv5x3x3_1/BiasAdd
B_conv5x3x3_1/ReluReluB_conv5x3x3_1/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2
B_conv5x3x3_1/ReluÑ
A_pool2_2/MaxPool3D	MaxPool3DA_conv3_2/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 *
ksize	
*
paddingVALID*
strides	
2
A_pool2_2/MaxPool3DÝ
B_pool4x2x2_1/MaxPool3D	MaxPool3D B_conv5x3x3_1/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2
B_pool4x2x2_1/MaxPool3D·
A_conv3_3/Conv3D/ReadVariableOpReadVariableOp(a_conv3_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
A_conv3_3/Conv3D/ReadVariableOpÝ
A_conv3_3/Conv3DConv3DA_pool2_2/MaxPool3D:output:0'A_conv3_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *
paddingVALID*
strides	
2
A_conv3_3/Conv3Dª
 A_conv3_3/BiasAdd/ReadVariableOpReadVariableOp)a_conv3_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 A_conv3_3/BiasAdd/ReadVariableOp´
A_conv3_3/BiasAddBiasAddA_conv3_3/Conv3D:output:0(A_conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
A_conv3_3/BiasAdd
A_conv3_3/ReluReluA_conv3_3/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
A_conv3_3/ReluÃ
#B_conv5x3x3_2/Conv3D/ReadVariableOpReadVariableOp,b_conv5x3x3_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02%
#B_conv5x3x3_2/Conv3D/ReadVariableOpí
B_conv5x3x3_2/Conv3DConv3D B_pool4x2x2_1/MaxPool3D:output:0+B_conv5x3x3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides	
2
B_conv5x3x3_2/Conv3D¶
$B_conv5x3x3_2/BiasAdd/ReadVariableOpReadVariableOp-b_conv5x3x3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$B_conv5x3x3_2/BiasAdd/ReadVariableOpÄ
B_conv5x3x3_2/BiasAddBiasAddB_conv5x3x3_2/Conv3D:output:0,B_conv5x3x3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
B_conv5x3x3_2/BiasAdd
B_conv5x3x3_2/ReluReluB_conv5x3x3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
B_conv5x3x3_2/Relu~
spatial_dropout3d/ShapeShapeA_conv3_3/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout3d/Shape
%spatial_dropout3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%spatial_dropout3d/strided_slice/stack
'spatial_dropout3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout3d/strided_slice/stack_1
'spatial_dropout3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout3d/strided_slice/stack_2Î
spatial_dropout3d/strided_sliceStridedSlice spatial_dropout3d/Shape:output:0.spatial_dropout3d/strided_slice/stack:output:00spatial_dropout3d/strided_slice/stack_1:output:00spatial_dropout3d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
spatial_dropout3d/strided_slice
'spatial_dropout3d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout3d/strided_slice_1/stack 
)spatial_dropout3d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout3d/strided_slice_1/stack_1 
)spatial_dropout3d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout3d/strided_slice_1/stack_2Ø
!spatial_dropout3d/strided_slice_1StridedSlice spatial_dropout3d/Shape:output:00spatial_dropout3d/strided_slice_1/stack:output:02spatial_dropout3d/strided_slice_1/stack_1:output:02spatial_dropout3d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout3d/strided_slice_1
spatial_dropout3d/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2!
spatial_dropout3d/dropout/ConstË
spatial_dropout3d/dropout/MulMulA_conv3_3/Relu:activations:0(spatial_dropout3d/dropout/Const:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
spatial_dropout3d/dropout/Mul¦
0spatial_dropout3d/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout3d/dropout/random_uniform/shape/1¦
0spatial_dropout3d/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout3d/dropout/random_uniform/shape/2¦
0spatial_dropout3d/dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout3d/dropout/random_uniform/shape/3
.spatial_dropout3d/dropout/random_uniform/shapePack(spatial_dropout3d/strided_slice:output:09spatial_dropout3d/dropout/random_uniform/shape/1:output:09spatial_dropout3d/dropout/random_uniform/shape/2:output:09spatial_dropout3d/dropout/random_uniform/shape/3:output:0*spatial_dropout3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:20
.spatial_dropout3d/dropout/random_uniform/shape
6spatial_dropout3d/dropout/random_uniform/RandomUniformRandomUniform7spatial_dropout3d/dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype028
6spatial_dropout3d/dropout/random_uniform/RandomUniform
(spatial_dropout3d/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2*
(spatial_dropout3d/dropout/GreaterEqual/y
&spatial_dropout3d/dropout/GreaterEqualGreaterEqual?spatial_dropout3d/dropout/random_uniform/RandomUniform:output:01spatial_dropout3d/dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&spatial_dropout3d/dropout/GreaterEqualÊ
spatial_dropout3d/dropout/CastCast*spatial_dropout3d/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
spatial_dropout3d/dropout/CastÎ
spatial_dropout3d/dropout/Mul_1Mul!spatial_dropout3d/dropout/Mul:z:0"spatial_dropout3d/dropout/Cast:y:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2!
spatial_dropout3d/dropout/Mul_1
spatial_dropout3d_1/ShapeShape B_conv5x3x3_2/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout3d_1/Shape
'spatial_dropout3d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'spatial_dropout3d_1/strided_slice/stack 
)spatial_dropout3d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout3d_1/strided_slice/stack_1 
)spatial_dropout3d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout3d_1/strided_slice/stack_2Ú
!spatial_dropout3d_1/strided_sliceStridedSlice"spatial_dropout3d_1/Shape:output:00spatial_dropout3d_1/strided_slice/stack:output:02spatial_dropout3d_1/strided_slice/stack_1:output:02spatial_dropout3d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout3d_1/strided_slice 
)spatial_dropout3d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout3d_1/strided_slice_1/stack¤
+spatial_dropout3d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout3d_1/strided_slice_1/stack_1¤
+spatial_dropout3d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout3d_1/strided_slice_1/stack_2ä
#spatial_dropout3d_1/strided_slice_1StridedSlice"spatial_dropout3d_1/Shape:output:02spatial_dropout3d_1/strided_slice_1/stack:output:04spatial_dropout3d_1/strided_slice_1/stack_1:output:04spatial_dropout3d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#spatial_dropout3d_1/strided_slice_1
!spatial_dropout3d_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2#
!spatial_dropout3d_1/dropout/ConstÕ
spatial_dropout3d_1/dropout/MulMul B_conv5x3x3_2/Relu:activations:0*spatial_dropout3d_1/dropout/Const:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2!
spatial_dropout3d_1/dropout/Mulª
2spatial_dropout3d_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout3d_1/dropout/random_uniform/shape/1ª
2spatial_dropout3d_1/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout3d_1/dropout/random_uniform/shape/2ª
2spatial_dropout3d_1/dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout3d_1/dropout/random_uniform/shape/3«
0spatial_dropout3d_1/dropout/random_uniform/shapePack*spatial_dropout3d_1/strided_slice:output:0;spatial_dropout3d_1/dropout/random_uniform/shape/1:output:0;spatial_dropout3d_1/dropout/random_uniform/shape/2:output:0;spatial_dropout3d_1/dropout/random_uniform/shape/3:output:0,spatial_dropout3d_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:22
0spatial_dropout3d_1/dropout/random_uniform/shape
8spatial_dropout3d_1/dropout/random_uniform/RandomUniformRandomUniform9spatial_dropout3d_1/dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02:
8spatial_dropout3d_1/dropout/random_uniform/RandomUniform
*spatial_dropout3d_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2,
*spatial_dropout3d_1/dropout/GreaterEqual/y£
(spatial_dropout3d_1/dropout/GreaterEqualGreaterEqualAspatial_dropout3d_1/dropout/random_uniform/RandomUniform:output:03spatial_dropout3d_1/dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(spatial_dropout3d_1/dropout/GreaterEqualÐ
 spatial_dropout3d_1/dropout/CastCast,spatial_dropout3d_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 spatial_dropout3d_1/dropout/CastÖ
!spatial_dropout3d_1/dropout/Mul_1Mul#spatial_dropout3d_1/dropout/Mul:z:0$spatial_dropout3d_1/dropout/Cast:y:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2#
!spatial_dropout3d_1/dropout/Mul_1Ø
A_pool2_3/MaxPool3D	MaxPool3D#spatial_dropout3d/dropout/Mul_1:z:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
ksize	
*
paddingVALID*
strides	
2
A_pool2_3/MaxPool3Dk
A_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
A_out/Const
A_out/ReshapeReshapeA_pool2_3/MaxPool3D:output:0A_out/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
A_out/Reshapek
B_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
B_out/Const
B_out/ReshapeReshape%spatial_dropout3d_1/dropout/Mul_1:z:0B_out/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
B_out/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÂ
concatenate/concatConcatV2A_out/Reshape:output:0B_out/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
concatenate/concats
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/dropout/Const¡
dropout/dropout/MulMulconcatenate/concat:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/dropout/Muly
dropout/dropout/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÍ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yß
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/dropout/Mul_1¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
+*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_1/dropout/Const¤
dropout_1/dropout/MulMuldense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeÓ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_1/dropout/GreaterEqual/yç
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Cast£
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul_1§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp¡
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ:::::::::::::::::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ý
N
2__inference_spatial_dropout3d_layer_call_fn_632167

inputs
identityà
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6312032
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ. :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 
 
_user_specified_nameinputs
Ì
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_632364

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
F
*__inference_A_pool2_1_layer_call_fn_630844

inputs
identityü
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_6308382
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
a
E__inference_B_celpool_layer_call_and_return_conditional_losses_630850

inputs
identityË
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2
	AvgPool3D
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬	
±
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_631159

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

k
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632157

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2

Identity_1"!

identity_1Identity_1:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ. :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 
 
_user_specified_nameinputs
¬	
±
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_631103

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*
paddingVALID*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ(:::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Õ

Ú
-__inference_functional_3_layer_call_fn_631591
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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_6315562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
æ
J
.__inference_B_pool4x2x2_1_layer_call_fn_630880

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_6308742
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

k
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632118

inputs

identity_1
IdentityIdentityinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
]
A__inference_A_out_layer_call_and_return_conditional_losses_631262

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â
a
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_630862

inputs
identityË
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
e
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_630874

inputs
identityË
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®Q
ç
H__inference_functional_3_layer_call_and_return_conditional_losses_631649

inputs
a_conv3_1_631596
a_conv3_1_631598
a_conv3_2_631603
a_conv3_2_631605
b_conv5x3x3_1_631608
b_conv5x3x3_1_631610
a_conv3_3_631615
a_conv3_3_631617
b_conv5x3x3_2_631620
b_conv5x3x3_2_631622
dense_631632
dense_631634
dense_1_631638
dense_1_631640
dense_2_631643
dense_2_631645
identity¢!A_conv3_1/StatefulPartitionedCall¢!A_conv3_2/StatefulPartitionedCall¢!A_conv3_3/StatefulPartitionedCall¢%B_conv5x3x3_1/StatefulPartitionedCall¢%B_conv5x3x3_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¯
!A_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinputsa_conv3_1_631596a_conv3_1_631598*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_6310472#
!A_conv3_1/StatefulPartitionedCall
A_pool2_1/PartitionedCallPartitionedCall*A_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_6308382
A_pool2_1/PartitionedCallì
B_celpool/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_B_celpool_layer_call_and_return_conditional_losses_6308502
B_celpool/PartitionedCallÊ
!A_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_1/PartitionedCall:output:0a_conv3_2_631603a_conv3_2_631605*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_6310762#
!A_conv3_2/StatefulPartitionedCallÞ
%B_conv5x3x3_1/StatefulPartitionedCallStatefulPartitionedCall"B_celpool/PartitionedCall:output:0b_conv5x3x3_1_631608b_conv5x3x3_1_631610*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_6311032'
%B_conv5x3x3_1/StatefulPartitionedCall
A_pool2_2/PartitionedCallPartitionedCall*A_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_6308622
A_pool2_2/PartitionedCall 
B_pool4x2x2_1/PartitionedCallPartitionedCall.B_conv5x3x3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_6308742
B_pool4x2x2_1/PartitionedCallÊ
!A_conv3_3/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_2/PartitionedCall:output:0a_conv3_3_631615a_conv3_3_631617*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_6311322#
!A_conv3_3/StatefulPartitionedCallâ
%B_conv5x3x3_2/StatefulPartitionedCallStatefulPartitionedCall&B_pool4x2x2_1/PartitionedCall:output:0b_conv5x3x3_2_631620b_conv5x3x3_2_631622*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_6311592'
%B_conv5x3x3_2/StatefulPartitionedCall¨
!spatial_dropout3d/PartitionedCallPartitionedCall*A_conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6312032#
!spatial_dropout3d/PartitionedCall²
#spatial_dropout3d_1/PartitionedCallPartitionedCall.B_conv5x3x3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6312432%
#spatial_dropout3d_1/PartitionedCall
A_pool2_3/PartitionedCallPartitionedCall*spatial_dropout3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_6309562
A_pool2_3/PartitionedCallñ
A_out/PartitionedCallPartitionedCall"A_pool2_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_A_out_layer_call_and_return_conditional_losses_6312622
A_out/PartitionedCallû
B_out/PartitionedCallPartitionedCall,spatial_dropout3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_B_out_layer_call_and_return_conditional_losses_6312762
B_out/PartitionedCall 
concatenate/PartitionedCallPartitionedCallA_out/PartitionedCall:output:0B_out/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6312912
concatenate/PartitionedCallù
dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6313172
dropout/PartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_631632dense_631634*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6313412
dense/StatefulPartitionedCall
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6313742
dropout_1/PartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_631638dense_1_631640*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6313982!
dense_1/StatefulPartitionedCallº
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_631643dense_2_631645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6314242!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^A_conv3_1/StatefulPartitionedCall"^A_conv3_2/StatefulPartitionedCall"^A_conv3_3/StatefulPartitionedCall&^B_conv5x3x3_1/StatefulPartitionedCall&^B_conv5x3x3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::2F
!A_conv3_1/StatefulPartitionedCall!A_conv3_1/StatefulPartitionedCall2F
!A_conv3_2/StatefulPartitionedCall!A_conv3_2/StatefulPartitionedCall2F
!A_conv3_3/StatefulPartitionedCall!A_conv3_3/StatefulPartitionedCall2N
%B_conv5x3x3_1/StatefulPartitionedCall%B_conv5x3x3_1/StatefulPartitionedCall2N
%B_conv5x3x3_2/StatefulPartitionedCall%B_conv5x3x3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ï
«
C__inference_dense_2_layer_call_and_return_conditional_losses_631424

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
a
C__inference_dropout_layer_call_and_return_conditional_losses_631317

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs
Ò

Ù
-__inference_functional_3_layer_call_fn_632009

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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_6316492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs

m
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632255

inputs

identity_1
IdentityIdentityinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨	
­
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_632040

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *
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
:ÿÿÿÿÿÿÿÿÿa 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿc :::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc 
 
_user_specified_nameinputs
¯
B
&__inference_A_out_layer_call_fn_632276

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_A_out_layer_call_and_return_conditional_losses_6312622
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×ï
ä
"__inference__traced_restore_632788
file_prefix%
!assignvariableop_a_conv3_1_kernel%
!assignvariableop_1_a_conv3_1_bias'
#assignvariableop_2_a_conv3_2_kernel%
!assignvariableop_3_a_conv3_2_bias+
'assignvariableop_4_b_conv5x3x3_1_kernel)
%assignvariableop_5_b_conv5x3x3_1_bias'
#assignvariableop_6_a_conv3_3_kernel%
!assignvariableop_7_a_conv3_3_bias+
'assignvariableop_8_b_conv5x3x3_2_kernel)
%assignvariableop_9_b_conv5x3x3_2_bias$
 assignvariableop_10_dense_kernel"
assignvariableop_11_dense_bias&
"assignvariableop_12_dense_1_kernel$
 assignvariableop_13_dense_1_bias&
"assignvariableop_14_dense_2_kernel$
 assignvariableop_15_dense_2_bias!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1/
+assignvariableop_25_adam_a_conv3_1_kernel_m-
)assignvariableop_26_adam_a_conv3_1_bias_m/
+assignvariableop_27_adam_a_conv3_2_kernel_m-
)assignvariableop_28_adam_a_conv3_2_bias_m3
/assignvariableop_29_adam_b_conv5x3x3_1_kernel_m1
-assignvariableop_30_adam_b_conv5x3x3_1_bias_m/
+assignvariableop_31_adam_a_conv3_3_kernel_m-
)assignvariableop_32_adam_a_conv3_3_bias_m3
/assignvariableop_33_adam_b_conv5x3x3_2_kernel_m1
-assignvariableop_34_adam_b_conv5x3x3_2_bias_m+
'assignvariableop_35_adam_dense_kernel_m)
%assignvariableop_36_adam_dense_bias_m-
)assignvariableop_37_adam_dense_1_kernel_m+
'assignvariableop_38_adam_dense_1_bias_m-
)assignvariableop_39_adam_dense_2_kernel_m+
'assignvariableop_40_adam_dense_2_bias_m/
+assignvariableop_41_adam_a_conv3_1_kernel_v-
)assignvariableop_42_adam_a_conv3_1_bias_v/
+assignvariableop_43_adam_a_conv3_2_kernel_v-
)assignvariableop_44_adam_a_conv3_2_bias_v3
/assignvariableop_45_adam_b_conv5x3x3_1_kernel_v1
-assignvariableop_46_adam_b_conv5x3x3_1_bias_v/
+assignvariableop_47_adam_a_conv3_3_kernel_v-
)assignvariableop_48_adam_a_conv3_3_bias_v3
/assignvariableop_49_adam_b_conv5x3x3_2_kernel_v1
-assignvariableop_50_adam_b_conv5x3x3_2_bias_v+
'assignvariableop_51_adam_dense_kernel_v)
%assignvariableop_52_adam_dense_bias_v-
)assignvariableop_53_adam_dense_1_kernel_v+
'assignvariableop_54_adam_dense_1_bias_v-
)assignvariableop_55_adam_dense_2_kernel_v+
'assignvariableop_56_adam_dense_2_bias_v
identity_58¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9´ 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*À
value¶B³:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÐ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*þ
_output_shapesë
è::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_a_conv3_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_a_conv3_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_a_conv3_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_a_conv3_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_b_conv5x3x3_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ª
AssignVariableOp_5AssignVariableOp%assignvariableop_5_b_conv5x3x3_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_a_conv3_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_a_conv3_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¬
AssignVariableOp_8AssignVariableOp'assignvariableop_8_b_conv5x3x3_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ª
AssignVariableOp_9AssignVariableOp%assignvariableop_9_b_conv5x3x3_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¨
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¦
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ª
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¨
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¨
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¥
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17§
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18§
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¦
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¡
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23£
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24£
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25³
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_a_conv3_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26±
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_a_conv3_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_a_conv3_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_a_conv3_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29·
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_b_conv5x3x3_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30µ
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_b_conv5x3x3_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_a_conv3_3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_a_conv3_3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33·
AssignVariableOp_33AssignVariableOp/assignvariableop_33_adam_b_conv5x3x3_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34µ
AssignVariableOp_34AssignVariableOp-assignvariableop_34_adam_b_conv5x3x3_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¯
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36­
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37±
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¯
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¯
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41³
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_a_conv3_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42±
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_a_conv3_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43³
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_a_conv3_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_a_conv3_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45·
AssignVariableOp_45AssignVariableOp/assignvariableop_45_adam_b_conv5x3x3_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46µ
AssignVariableOp_46AssignVariableOp-assignvariableop_46_adam_b_conv5x3x3_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47³
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_a_conv3_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_a_conv3_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49·
AssignVariableOp_49AssignVariableOp/assignvariableop_49_adam_b_conv5x3x3_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50µ
AssignVariableOp_50AssignVariableOp-assignvariableop_50_adam_b_conv5x3x3_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¯
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52­
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_dense_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53±
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¯
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55±
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¯
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_569
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÄ

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_57·

Identity_58IdentityIdentity_57:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_58"#
identity_58Identity_58:output:0*û
_input_shapesé
æ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
®	
­
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_631047

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOpª
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *
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
 :ÿÿÿÿÿÿÿÿÿÆ 2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ù
k
2__inference_spatial_dropout3d_layer_call_fn_632123

inputs
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6309372
StatefulPartitionedCall¾
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
]
A__inference_A_out_layer_call_and_return_conditional_losses_632271

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
²Q
è
H__inference_functional_3_layer_call_and_return_conditional_losses_631497
input_1
a_conv3_1_631444
a_conv3_1_631446
a_conv3_2_631451
a_conv3_2_631453
b_conv5x3x3_1_631456
b_conv5x3x3_1_631458
a_conv3_3_631463
a_conv3_3_631465
b_conv5x3x3_2_631468
b_conv5x3x3_2_631470
dense_631480
dense_631482
dense_1_631486
dense_1_631488
dense_2_631491
dense_2_631493
identity¢!A_conv3_1/StatefulPartitionedCall¢!A_conv3_2/StatefulPartitionedCall¢!A_conv3_3/StatefulPartitionedCall¢%B_conv5x3x3_1/StatefulPartitionedCall¢%B_conv5x3x3_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall°
!A_conv3_1/StatefulPartitionedCallStatefulPartitionedCallinput_1a_conv3_1_631444a_conv3_1_631446*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_6310472#
!A_conv3_1/StatefulPartitionedCall
A_pool2_1/PartitionedCallPartitionedCall*A_conv3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_6308382
A_pool2_1/PartitionedCallí
B_celpool/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_B_celpool_layer_call_and_return_conditional_losses_6308502
B_celpool/PartitionedCallÊ
!A_conv3_2/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_1/PartitionedCall:output:0a_conv3_2_631451a_conv3_2_631453*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_6310762#
!A_conv3_2/StatefulPartitionedCallÞ
%B_conv5x3x3_1/StatefulPartitionedCallStatefulPartitionedCall"B_celpool/PartitionedCall:output:0b_conv5x3x3_1_631456b_conv5x3x3_1_631458*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_6311032'
%B_conv5x3x3_1/StatefulPartitionedCall
A_pool2_2/PartitionedCallPartitionedCall*A_conv3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_6308622
A_pool2_2/PartitionedCall 
B_pool4x2x2_1/PartitionedCallPartitionedCall.B_conv5x3x3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_6308742
B_pool4x2x2_1/PartitionedCallÊ
!A_conv3_3/StatefulPartitionedCallStatefulPartitionedCall"A_pool2_2/PartitionedCall:output:0a_conv3_3_631463a_conv3_3_631465*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_6311322#
!A_conv3_3/StatefulPartitionedCallâ
%B_conv5x3x3_2/StatefulPartitionedCallStatefulPartitionedCall&B_pool4x2x2_1/PartitionedCall:output:0b_conv5x3x3_2_631468b_conv5x3x3_2_631470*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_6311592'
%B_conv5x3x3_2/StatefulPartitionedCall¨
!spatial_dropout3d/PartitionedCallPartitionedCall*A_conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_6312032#
!spatial_dropout3d/PartitionedCall²
#spatial_dropout3d_1/PartitionedCallPartitionedCall.B_conv5x3x3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6312432%
#spatial_dropout3d_1/PartitionedCall
A_pool2_3/PartitionedCallPartitionedCall*spatial_dropout3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_6309562
A_pool2_3/PartitionedCallñ
A_out/PartitionedCallPartitionedCall"A_pool2_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_A_out_layer_call_and_return_conditional_losses_6312622
A_out/PartitionedCallû
B_out/PartitionedCallPartitionedCall,spatial_dropout3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_B_out_layer_call_and_return_conditional_losses_6312762
B_out/PartitionedCall 
concatenate/PartitionedCallPartitionedCallA_out/PartitionedCall:output:0B_out/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6312912
concatenate/PartitionedCallù
dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6313172
dropout/PartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_631480dense_631482*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6313412
dense/StatefulPartitionedCall
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6313742
dropout_1/PartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_631486dense_1_631488*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6313982!
dense_1/StatefulPartitionedCallº
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_631491dense_2_631493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6314242!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^A_conv3_1/StatefulPartitionedCall"^A_conv3_2/StatefulPartitionedCall"^A_conv3_3/StatefulPartitionedCall&^B_conv5x3x3_1/StatefulPartitionedCall&^B_conv5x3x3_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ::::::::::::::::2F
!A_conv3_1/StatefulPartitionedCall!A_conv3_1/StatefulPartitionedCall2F
!A_conv3_2/StatefulPartitionedCall!A_conv3_2/StatefulPartitionedCall2F
!A_conv3_3/StatefulPartitionedCall!A_conv3_3/StatefulPartitionedCall2N
%B_conv5x3x3_1/StatefulPartitionedCall%B_conv5x3x3_1/StatefulPartitionedCall2N
%B_conv5x3x3_2/StatefulPartitionedCall%B_conv5x3x3_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1

b
C__inference_dropout_layer_call_and_return_conditional_losses_632312

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs
¬	
±
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_632178

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
C__inference_dropout_layer_call_and_return_conditional_losses_631312

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ+:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
 
_user_specified_nameinputs
á
P
4__inference_spatial_dropout3d_1_layer_call_fn_632226

inputs
identityâ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *X
fSRQ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_6312432
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬	
±
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_632060

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*
paddingVALID*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ(:::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_632359

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨	
­
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_631076

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp©
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *
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
:ÿÿÿÿÿÿÿÿÿa 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2
Relur
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿc :::[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc 
 
_user_specified_nameinputs
¸e
ù
!__inference__wrapped_model_630832
input_19
5functional_3_a_conv3_1_conv3d_readvariableop_resource:
6functional_3_a_conv3_1_biasadd_readvariableop_resource9
5functional_3_a_conv3_2_conv3d_readvariableop_resource:
6functional_3_a_conv3_2_biasadd_readvariableop_resource=
9functional_3_b_conv5x3x3_1_conv3d_readvariableop_resource>
:functional_3_b_conv5x3x3_1_biasadd_readvariableop_resource9
5functional_3_a_conv3_3_conv3d_readvariableop_resource:
6functional_3_a_conv3_3_biasadd_readvariableop_resource=
9functional_3_b_conv5x3x3_2_conv3d_readvariableop_resource>
:functional_3_b_conv5x3x3_2_biasadd_readvariableop_resource5
1functional_3_dense_matmul_readvariableop_resource6
2functional_3_dense_biasadd_readvariableop_resource7
3functional_3_dense_1_matmul_readvariableop_resource8
4functional_3_dense_1_biasadd_readvariableop_resource7
3functional_3_dense_2_matmul_readvariableop_resource8
4functional_3_dense_2_biasadd_readvariableop_resource
identityÞ
,functional_3/A_conv3_1/Conv3D/ReadVariableOpReadVariableOp5functional_3_a_conv3_1_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02.
,functional_3/A_conv3_1/Conv3D/ReadVariableOpð
functional_3/A_conv3_1/Conv3DConv3Dinput_14functional_3/A_conv3_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ *
paddingVALID*
strides	
2
functional_3/A_conv3_1/Conv3DÑ
-functional_3/A_conv3_1/BiasAdd/ReadVariableOpReadVariableOp6functional_3_a_conv3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-functional_3/A_conv3_1/BiasAdd/ReadVariableOpé
functional_3/A_conv3_1/BiasAddBiasAdd&functional_3/A_conv3_1/Conv3D:output:05functional_3/A_conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2 
functional_3/A_conv3_1/BiasAddª
functional_3/A_conv3_1/ReluRelu'functional_3/A_conv3_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÆ 2
functional_3/A_conv3_1/Reluø
 functional_3/A_pool2_1/MaxPool3D	MaxPool3D)functional_3/A_conv3_1/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿc *
ksize	
*
paddingVALID*
strides	
2"
 functional_3/A_pool2_1/MaxPool3DÖ
 functional_3/B_celpool/AvgPool3D	AvgPool3Dinput_1*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ(*
ksize	
*
paddingVALID*
strides	
2"
 functional_3/B_celpool/AvgPool3DÞ
,functional_3/A_conv3_2/Conv3D/ReadVariableOpReadVariableOp5functional_3_a_conv3_2_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02.
,functional_3/A_conv3_2/Conv3D/ReadVariableOp
functional_3/A_conv3_2/Conv3DConv3D)functional_3/A_pool2_1/MaxPool3D:output:04functional_3/A_conv3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa *
paddingVALID*
strides	
2
functional_3/A_conv3_2/Conv3DÑ
-functional_3/A_conv3_2/BiasAdd/ReadVariableOpReadVariableOp6functional_3_a_conv3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-functional_3/A_conv3_2/BiasAdd/ReadVariableOpè
functional_3/A_conv3_2/BiasAddBiasAdd&functional_3/A_conv3_2/Conv3D:output:05functional_3/A_conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2 
functional_3/A_conv3_2/BiasAdd©
functional_3/A_conv3_2/ReluRelu'functional_3/A_conv3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿa 2
functional_3/A_conv3_2/Reluê
0functional_3/B_conv5x3x3_1/Conv3D/ReadVariableOpReadVariableOp9functional_3_b_conv5x3x3_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype022
0functional_3/B_conv5x3x3_1/Conv3D/ReadVariableOp
!functional_3/B_conv5x3x3_1/Conv3DConv3D)functional_3/B_celpool/AvgPool3D:output:08functional_3/B_conv5x3x3_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$*
paddingVALID*
strides	
2#
!functional_3/B_conv5x3x3_1/Conv3DÝ
1functional_3/B_conv5x3x3_1/BiasAdd/ReadVariableOpReadVariableOp:functional_3_b_conv5x3x3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_3/B_conv5x3x3_1/BiasAdd/ReadVariableOpø
"functional_3/B_conv5x3x3_1/BiasAddBiasAdd*functional_3/B_conv5x3x3_1/Conv3D:output:09functional_3/B_conv5x3x3_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2$
"functional_3/B_conv5x3x3_1/BiasAddµ
functional_3/B_conv5x3x3_1/ReluRelu+functional_3/B_conv5x3x3_1/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ$2!
functional_3/B_conv5x3x3_1/Reluø
 functional_3/A_pool2_2/MaxPool3D	MaxPool3D)functional_3/A_conv3_2/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ0 *
ksize	
*
paddingVALID*
strides	
2"
 functional_3/A_pool2_2/MaxPool3D
$functional_3/B_pool4x2x2_1/MaxPool3D	MaxPool3D-functional_3/B_conv5x3x3_1/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2&
$functional_3/B_pool4x2x2_1/MaxPool3DÞ
,functional_3/A_conv3_3/Conv3D/ReadVariableOpReadVariableOp5functional_3_a_conv3_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02.
,functional_3/A_conv3_3/Conv3D/ReadVariableOp
functional_3/A_conv3_3/Conv3DConv3D)functional_3/A_pool2_2/MaxPool3D:output:04functional_3/A_conv3_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. *
paddingVALID*
strides	
2
functional_3/A_conv3_3/Conv3DÑ
-functional_3/A_conv3_3/BiasAdd/ReadVariableOpReadVariableOp6functional_3_a_conv3_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-functional_3/A_conv3_3/BiasAdd/ReadVariableOpè
functional_3/A_conv3_3/BiasAddBiasAdd&functional_3/A_conv3_3/Conv3D:output:05functional_3/A_conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2 
functional_3/A_conv3_3/BiasAdd©
functional_3/A_conv3_3/ReluRelu'functional_3/A_conv3_3/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2
functional_3/A_conv3_3/Reluê
0functional_3/B_conv5x3x3_2/Conv3D/ReadVariableOpReadVariableOp9functional_3_b_conv5x3x3_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype022
0functional_3/B_conv5x3x3_2/Conv3D/ReadVariableOp¡
!functional_3/B_conv5x3x3_2/Conv3DConv3D-functional_3/B_pool4x2x2_1/MaxPool3D:output:08functional_3/B_conv5x3x3_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides	
2#
!functional_3/B_conv5x3x3_2/Conv3DÝ
1functional_3/B_conv5x3x3_2/BiasAdd/ReadVariableOpReadVariableOp:functional_3_b_conv5x3x3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1functional_3/B_conv5x3x3_2/BiasAdd/ReadVariableOpø
"functional_3/B_conv5x3x3_2/BiasAddBiasAdd*functional_3/B_conv5x3x3_2/Conv3D:output:09functional_3/B_conv5x3x3_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2$
"functional_3/B_conv5x3x3_2/BiasAddµ
functional_3/B_conv5x3x3_2/ReluRelu+functional_3/B_conv5x3x3_2/BiasAdd:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2!
functional_3/B_conv5x3x3_2/ReluÇ
'functional_3/spatial_dropout3d/IdentityIdentity)functional_3/A_conv3_3/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ. 2)
'functional_3/spatial_dropout3d/IdentityÏ
)functional_3/spatial_dropout3d_1/IdentityIdentity-functional_3/B_conv5x3x3_2/Relu:activations:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2+
)functional_3/spatial_dropout3d_1/Identityÿ
 functional_3/A_pool2_3/MaxPool3D	MaxPool3D0functional_3/spatial_dropout3d/Identity:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *
ksize	
*
paddingVALID*
strides	
2"
 functional_3/A_pool2_3/MaxPool3D
functional_3/A_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
functional_3/A_out/ConstÄ
functional_3/A_out/ReshapeReshape)functional_3/A_pool2_3/MaxPool3D:output:0!functional_3/A_out/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/A_out/Reshape
functional_3/B_out/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
functional_3/B_out/ConstÍ
functional_3/B_out/ReshapeReshape2functional_3/spatial_dropout3d_1/Identity:output:0!functional_3/B_out/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_3/B_out/Reshape
$functional_3/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_3/concatenate/concat/axis
functional_3/concatenate/concatConcatV2#functional_3/A_out/Reshape:output:0#functional_3/B_out/Reshape:output:0-functional_3/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2!
functional_3/concatenate/concat§
functional_3/dropout/IdentityIdentity(functional_3/concatenate/concat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
functional_3/dropout/IdentityÈ
(functional_3/dense/MatMul/ReadVariableOpReadVariableOp1functional_3_dense_matmul_readvariableop_resource* 
_output_shapes
:
+*
dtype02*
(functional_3/dense/MatMul/ReadVariableOpÍ
functional_3/dense/MatMulMatMul&functional_3/dropout/Identity:output:00functional_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense/MatMulÆ
)functional_3/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)functional_3/dense/BiasAdd/ReadVariableOpÎ
functional_3/dense/BiasAddBiasAdd#functional_3/dense/MatMul:product:01functional_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense/BiasAdd
functional_3/dense/ReluRelu#functional_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense/Relu¨
functional_3/dropout_1/IdentityIdentity%functional_3/dense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
functional_3/dropout_1/IdentityÎ
*functional_3/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_3_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*functional_3/dense_1/MatMul/ReadVariableOpÕ
functional_3/dense_1/MatMulMatMul(functional_3/dropout_1/Identity:output:02functional_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense_1/MatMulÌ
+functional_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_3/dense_1/BiasAdd/ReadVariableOpÖ
functional_3/dense_1/BiasAddBiasAdd%functional_3/dense_1/MatMul:product:03functional_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense_1/BiasAdd
functional_3/dense_1/ReluRelu%functional_3/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense_1/ReluÍ
*functional_3/dense_2/MatMul/ReadVariableOpReadVariableOp3functional_3_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*functional_3/dense_2/MatMul/ReadVariableOpÓ
functional_3/dense_2/MatMulMatMul'functional_3/dense_1/Relu:activations:02functional_3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense_2/MatMulË
+functional_3/dense_2/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_3/dense_2/BiasAdd/ReadVariableOpÕ
functional_3/dense_2/BiasAddBiasAdd%functional_3/dense_2/MatMul:product:03functional_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_3/dense_2/BiasAddy
IdentityIdentity%functional_3/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÈ:::::::::::::::::] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
º
q
G__inference_concatenate_layer_call_and_return_conditional_losses_631291

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â
a
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_630956

inputs
identityË
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: {
W
_output_shapesE
C:Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
n
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632211

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2
dropout/random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/3
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0'dropout/random_uniform/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeØ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/yÓ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â
s
G__inference_concatenate_layer_call_and_return_conditional_losses_632294
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1"¸L
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
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿÈ;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:±á
Ã
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
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-5
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"
_tf_keras_networkë{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "A_conv3_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "A_conv3_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "A_pool2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "A_pool2_1", "inbound_nodes": [[["A_conv3_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "A_conv3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "A_conv3_2", "inbound_nodes": [[["A_pool2_1", 0, 0, {}]]]}, {"class_name": "AveragePooling3D", "config": {"name": "B_celpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 1, 1]}, "data_format": "channels_last"}, "name": "B_celpool", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "A_pool2_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "A_pool2_2", "inbound_nodes": [[["A_conv3_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "B_conv5x3x3_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv5x3x3_1", "inbound_nodes": [[["B_celpool", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "A_conv3_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "A_conv3_3", "inbound_nodes": [[["A_pool2_2", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "B_pool4x2x2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3, 3]}, "data_format": "channels_last"}, "name": "B_pool4x2x2_1", "inbound_nodes": [[["B_conv5x3x3_1", 0, 0, {}]]]}, {"class_name": "SpatialDropout3D", "config": {"name": "spatial_dropout3d", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "spatial_dropout3d", "inbound_nodes": [[["A_conv3_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "B_conv5x3x3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv5x3x3_2", "inbound_nodes": [[["B_pool4x2x2_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "A_pool2_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 2, 2]}, "data_format": "channels_last"}, "name": "A_pool2_3", "inbound_nodes": [[["spatial_dropout3d", 0, 0, {}]]]}, {"class_name": "SpatialDropout3D", "config": {"name": "spatial_dropout3d_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "spatial_dropout3d_1", "inbound_nodes": [[["B_conv5x3x3_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "A_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "A_out", "inbound_nodes": [[["A_pool2_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "B_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "B_out", "inbound_nodes": [[["spatial_dropout3d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate", "inbound_nodes": [[["A_out", 0, 0, {}], ["B_out", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 30, 30, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "A_conv3_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "A_conv3_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "A_pool2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "A_pool2_1", "inbound_nodes": [[["A_conv3_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "A_conv3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "A_conv3_2", "inbound_nodes": [[["A_pool2_1", 0, 0, {}]]]}, {"class_name": "AveragePooling3D", "config": {"name": "B_celpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 1, 1]}, "data_format": "channels_last"}, "name": "B_celpool", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "A_pool2_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "A_pool2_2", "inbound_nodes": [[["A_conv3_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "B_conv5x3x3_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv5x3x3_1", "inbound_nodes": [[["B_celpool", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "A_conv3_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "A_conv3_3", "inbound_nodes": [[["A_pool2_2", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "B_pool4x2x2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3, 3]}, "data_format": "channels_last"}, "name": "B_pool4x2x2_1", "inbound_nodes": [[["B_conv5x3x3_1", 0, 0, {}]]]}, {"class_name": "SpatialDropout3D", "config": {"name": "spatial_dropout3d", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "spatial_dropout3d", "inbound_nodes": [[["A_conv3_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "B_conv5x3x3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "B_conv5x3x3_2", "inbound_nodes": [[["B_pool4x2x2_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "A_pool2_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 2, 2]}, "data_format": "channels_last"}, "name": "A_pool2_3", "inbound_nodes": [[["spatial_dropout3d", 0, 0, {}]]]}, {"class_name": "SpatialDropout3D", "config": {"name": "spatial_dropout3d_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "spatial_dropout3d_1", "inbound_nodes": [[["B_conv5x3x3_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "A_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "A_out", "inbound_nodes": [[["A_pool2_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "B_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "B_out", "inbound_nodes": [[["spatial_dropout3d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate", "inbound_nodes": [[["A_out", 0, 0, {}], ["B_out", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": "loss_fn", "metrics": ["mae"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"
_tf_keras_input_layerà{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 30, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}



kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
__call__
+&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "Conv3D", "name": "A_conv3_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "A_conv3_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 30, 30, 1]}}
û
"	variables
#regularization_losses
$trainable_variables
%	keras_api
__call__
+&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "MaxPooling3D", "name": "A_pool2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "A_pool2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}



&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
__call__
+&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "Conv3D", "name": "A_conv3_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "A_conv3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 99, 14, 14, 32]}}
ÿ
,	variables
-regularization_losses
.trainable_variables
/	keras_api
__call__
+&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "AveragePooling3D", "name": "B_celpool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_celpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
û
0	variables
1regularization_losses
2trainable_variables
3	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "MaxPooling3D", "name": "A_pool2_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "A_pool2_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}



4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"ã
_tf_keras_layerÉ{"class_name": "Conv3D", "name": "B_conv5x3x3_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_conv5x3x3_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 30, 30, 1]}}



:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"class_name": "Conv3D", "name": "A_conv3_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "A_conv3_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 6, 6, 32]}}

@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "MaxPooling3D", "name": "B_pool4x2x2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_pool4x2x2_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}

D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layeræ{"class_name": "SpatialDropout3D", "name": "spatial_dropout3d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout3d", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}



Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"ã
_tf_keras_layerÉ{"class_name": "Conv3D", "name": "B_conv5x3x3_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_conv5x3x3_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 8, 8, 16]}}
û
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "MaxPooling3D", "name": "A_pool2_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "A_pool2_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}

R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "SpatialDropout3D", "name": "spatial_dropout3d_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout3d_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
à
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
°__call__
+±&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Flatten", "name": "A_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "A_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
à
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
²__call__
+³&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Flatten", "name": "B_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "B_out", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Î
^	variables
_regularization_losses
`trainable_variables
a	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"½
_tf_keras_layer£{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1408]}, {"class_name": "TensorShape", "items": [null, 4096]}]}
ã
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ó

fkernel
gbias
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5504}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5504]}}
ç
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
º__call__
+»&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
õ

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
õ

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}

|iter

}beta_1

~beta_2
	decay
learning_ratemõmö&m÷'mø4mù5mú:mû;müHmýImþfmÿgmpmqmvmwmvv&v'v4v5v:v;vHvIvfvgvpvqvvvwv"
	optimizer

0
1
&2
'3
44
55
:6
;7
H8
I9
f10
g11
p12
q13
v14
w15"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
&2
'3
44
55
:6
;7
H8
I9
f10
g11
p12
q13
v14
w15"
trackable_list_wrapper
Ó
 layer_regularization_losses
	variables
layer_metrics
regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Àserving_default"
signature_map
.:, 2A_conv3_1/kernel
: 2A_conv3_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
µ
 layer_regularization_losses
	variables
layer_metrics
regularization_losses
 trainable_variables
metrics
non_trainable_variables
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
"	variables
layer_metrics
#regularization_losses
$trainable_variables
metrics
non_trainable_variables
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,  2A_conv3_2/kernel
: 2A_conv3_2/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
µ
 layer_regularization_losses
(	variables
layer_metrics
)regularization_losses
*trainable_variables
metrics
non_trainable_variables
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
,	variables
layer_metrics
-regularization_losses
.trainable_variables
metrics
non_trainable_variables
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
0	variables
layer_metrics
1regularization_losses
2trainable_variables
metrics
non_trainable_variables
layers
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
2:02B_conv5x3x3_1/kernel
 :2B_conv5x3x3_1/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
µ
 layer_regularization_losses
6	variables
 layer_metrics
7regularization_losses
8trainable_variables
¡metrics
¢non_trainable_variables
£layers
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
.:,  2A_conv3_3/kernel
: 2A_conv3_3/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
µ
 ¤layer_regularization_losses
<	variables
¥layer_metrics
=regularization_losses
>trainable_variables
¦metrics
§non_trainable_variables
¨layers
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ©layer_regularization_losses
@	variables
ªlayer_metrics
Aregularization_losses
Btrainable_variables
«metrics
¬non_trainable_variables
­layers
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ®layer_regularization_losses
D	variables
¯layer_metrics
Eregularization_losses
Ftrainable_variables
°metrics
±non_trainable_variables
²layers
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
2:0 2B_conv5x3x3_2/kernel
 : 2B_conv5x3x3_2/bias
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
µ
 ³layer_regularization_losses
J	variables
´layer_metrics
Kregularization_losses
Ltrainable_variables
µmetrics
¶non_trainable_variables
·layers
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ¸layer_regularization_losses
N	variables
¹layer_metrics
Oregularization_losses
Ptrainable_variables
ºmetrics
»non_trainable_variables
¼layers
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ½layer_regularization_losses
R	variables
¾layer_metrics
Sregularization_losses
Ttrainable_variables
¿metrics
Ànon_trainable_variables
Álayers
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Âlayer_regularization_losses
V	variables
Ãlayer_metrics
Wregularization_losses
Xtrainable_variables
Ämetrics
Ånon_trainable_variables
Ælayers
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Çlayer_regularization_losses
Z	variables
Èlayer_metrics
[regularization_losses
\trainable_variables
Émetrics
Ênon_trainable_variables
Ëlayers
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Ìlayer_regularization_losses
^	variables
Ílayer_metrics
_regularization_losses
`trainable_variables
Îmetrics
Ïnon_trainable_variables
Ðlayers
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Ñlayer_regularization_losses
b	variables
Òlayer_metrics
cregularization_losses
dtrainable_variables
Ómetrics
Ônon_trainable_variables
Õlayers
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 :
+2dense/kernel
:2
dense/bias
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
µ
 Ölayer_regularization_losses
h	variables
×layer_metrics
iregularization_losses
jtrainable_variables
Ømetrics
Ùnon_trainable_variables
Úlayers
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Ûlayer_regularization_losses
l	variables
Ülayer_metrics
mregularization_losses
ntrainable_variables
Ýmetrics
Þnon_trainable_variables
ßlayers
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
µ
 àlayer_regularization_losses
r	variables
álayer_metrics
sregularization_losses
ttrainable_variables
âmetrics
ãnon_trainable_variables
älayers
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
:2dense_2/bias
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
µ
 ålayer_regularization_losses
x	variables
ælayer_metrics
yregularization_losses
ztrainable_variables
çmetrics
ènon_trainable_variables
élayers
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
ê0
ë1"
trackable_list_wrapper
 "
trackable_list_wrapper
¾
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
10
11
12
13
14
15
16
17
18
19
20"
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
¿

ìtotal

ícount
î	variables
ï	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ù

ðtotal

ñcount
ò
_fn_kwargs
ó	variables
ô	keras_api"­
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
:  (2total
:  (2count
0
ì0
í1"
trackable_list_wrapper
.
î	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ð0
ñ1"
trackable_list_wrapper
.
ó	variables"
_generic_user_object
3:1 2Adam/A_conv3_1/kernel/m
!: 2Adam/A_conv3_1/bias/m
3:1  2Adam/A_conv3_2/kernel/m
!: 2Adam/A_conv3_2/bias/m
7:52Adam/B_conv5x3x3_1/kernel/m
%:#2Adam/B_conv5x3x3_1/bias/m
3:1  2Adam/A_conv3_3/kernel/m
!: 2Adam/A_conv3_3/bias/m
7:5 2Adam/B_conv5x3x3_2/kernel/m
%:# 2Adam/B_conv5x3x3_2/bias/m
%:#
+2Adam/dense/kernel/m
:2Adam/dense/bias/m
':%
2Adam/dense_1/kernel/m
 :2Adam/dense_1/bias/m
&:$	2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
3:1 2Adam/A_conv3_1/kernel/v
!: 2Adam/A_conv3_1/bias/v
3:1  2Adam/A_conv3_2/kernel/v
!: 2Adam/A_conv3_2/bias/v
7:52Adam/B_conv5x3x3_1/kernel/v
%:#2Adam/B_conv5x3x3_1/bias/v
3:1  2Adam/A_conv3_3/kernel/v
!: 2Adam/A_conv3_3/bias/v
7:5 2Adam/B_conv5x3x3_2/kernel/v
%:# 2Adam/B_conv5x3x3_2/bias/v
%:#
+2Adam/dense/kernel/v
:2Adam/dense/bias/v
':%
2Adam/dense_1/kernel/v
 :2Adam/dense_1/bias/v
&:$	2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
2ÿ
-__inference_functional_3_layer_call_fn_631972
-__inference_functional_3_layer_call_fn_631684
-__inference_functional_3_layer_call_fn_631591
-__inference_functional_3_layer_call_fn_632009À
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
kwonlydefaultsª 
annotationsª *
 
ì2é
!__inference__wrapped_model_630832Ã
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
annotationsª *3¢0
.+
input_1ÿÿÿÿÿÿÿÿÿÈ
î2ë
H__inference_functional_3_layer_call_and_return_conditional_losses_631935
H__inference_functional_3_layer_call_and_return_conditional_losses_631861
H__inference_functional_3_layer_call_and_return_conditional_losses_631441
H__inference_functional_3_layer_call_and_return_conditional_losses_631497À
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
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_A_conv3_1_layer_call_fn_632029¢
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
annotationsª *
 
ï2ì
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_632020¢
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
annotationsª *
 
2
*__inference_A_pool2_1_layer_call_fn_630844í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º2·
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_630838í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_A_conv3_2_layer_call_fn_632049¢
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
annotationsª *
 
ï2ì
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_632040¢
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
annotationsª *
 
2
*__inference_B_celpool_layer_call_fn_630856í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º2·
E__inference_B_celpool_layer_call_and_return_conditional_losses_630850í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_A_pool2_2_layer_call_fn_630868í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º2·
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_630862í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_B_conv5x3x3_1_layer_call_fn_632069¢
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
annotationsª *
 
ó2ð
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_632060¢
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
annotationsª *
 
Ô2Ñ
*__inference_A_conv3_3_layer_call_fn_632089¢
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
annotationsª *
 
ï2ì
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_632080¢
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
annotationsª *
 
£2 
.__inference_B_pool4x2x2_1_layer_call_fn_630880í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¾2»
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_630874í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
2__inference_spatial_dropout3d_layer_call_fn_632128
2__inference_spatial_dropout3d_layer_call_fn_632123
2__inference_spatial_dropout3d_layer_call_fn_632162
2__inference_spatial_dropout3d_layer_call_fn_632167´
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
kwonlydefaultsª 
annotationsª *
 
ö2ó
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632113
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632118
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632157
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632152´
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
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
.__inference_B_conv5x3x3_2_layer_call_fn_632187¢
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
annotationsª *
 
ó2ð
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_632178¢
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
annotationsª *
 
2
*__inference_A_pool2_3_layer_call_fn_630962í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º2·
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_630956í
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
annotationsª *M¢J
HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
4__inference_spatial_dropout3d_1_layer_call_fn_632226
4__inference_spatial_dropout3d_1_layer_call_fn_632260
4__inference_spatial_dropout3d_1_layer_call_fn_632221
4__inference_spatial_dropout3d_1_layer_call_fn_632265´
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
kwonlydefaultsª 
annotationsª *
 
þ2û
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632255
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632216
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632250
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632211´
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
kwonlydefaultsª 
annotationsª *
 
Ð2Í
&__inference_A_out_layer_call_fn_632276¢
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
annotationsª *
 
ë2è
A__inference_A_out_layer_call_and_return_conditional_losses_632271¢
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
annotationsª *
 
Ð2Í
&__inference_B_out_layer_call_fn_632287¢
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
annotationsª *
 
ë2è
A__inference_B_out_layer_call_and_return_conditional_losses_632282¢
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
annotationsª *
 
Ö2Ó
,__inference_concatenate_layer_call_fn_632300¢
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
annotationsª *
 
ñ2î
G__inference_concatenate_layer_call_and_return_conditional_losses_632294¢
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
annotationsª *
 
2
(__inference_dropout_layer_call_fn_632327
(__inference_dropout_layer_call_fn_632322´
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
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_layer_call_and_return_conditional_losses_632317
C__inference_dropout_layer_call_and_return_conditional_losses_632312´
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
kwonlydefaultsª 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_632347¢
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
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_632338¢
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
annotationsª *
 
2
*__inference_dropout_1_layer_call_fn_632374
*__inference_dropout_1_layer_call_fn_632369´
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
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_1_layer_call_and_return_conditional_losses_632364
E__inference_dropout_1_layer_call_and_return_conditional_losses_632359´
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
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_632394¢
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
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_632385¢
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
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_632413¢
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
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_632404¢
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
annotationsª *
 
3B1
$__inference_signature_wrapper_631735input_1¿
E__inference_A_conv3_1_layer_call_and_return_conditional_losses_632020v<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÈ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÆ 
 
*__inference_A_conv3_1_layer_call_fn_632029i<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÈ
ª "%"ÿÿÿÿÿÿÿÿÿÆ ½
E__inference_A_conv3_2_layer_call_and_return_conditional_losses_632040t&';¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿc 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿa 
 
*__inference_A_conv3_2_layer_call_fn_632049g&';¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿc 
ª "$!ÿÿÿÿÿÿÿÿÿa ½
E__inference_A_conv3_3_layer_call_and_return_conditional_losses_632080t:;;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ0 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ. 
 
*__inference_A_conv3_3_layer_call_fn_632089g:;;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ0 
ª "$!ÿÿÿÿÿÿÿÿÿ. ª
A__inference_A_out_layer_call_and_return_conditional_losses_632271e;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_A_out_layer_call_fn_632276X;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ
E__inference_A_pool2_1_layer_call_and_return_conditional_losses_630838¸_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ú
*__inference_A_pool2_1_layer_call_fn_630844«_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
E__inference_A_pool2_2_layer_call_and_return_conditional_losses_630862¸_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ú
*__inference_A_pool2_2_layer_call_fn_630868«_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
E__inference_A_pool2_3_layer_call_and_return_conditional_losses_630956¸_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ú
*__inference_A_pool2_3_layer_call_fn_630962«_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
E__inference_B_celpool_layer_call_and_return_conditional_losses_630850¸_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ú
*__inference_B_celpool_layer_call_fn_630856«_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
I__inference_B_conv5x3x3_1_layer_call_and_return_conditional_losses_632060t45;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ(
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ$
 
.__inference_B_conv5x3x3_1_layer_call_fn_632069g45;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ(
ª "$!ÿÿÿÿÿÿÿÿÿ$Á
I__inference_B_conv5x3x3_2_layer_call_and_return_conditional_losses_632178tHI;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_B_conv5x3x3_2_layer_call_fn_632187gHI;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ
ª "$!ÿÿÿÿÿÿÿÿÿ ª
A__inference_B_out_layer_call_and_return_conditional_losses_632282e;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 
&__inference_B_out_layer_call_fn_632287X;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ 
I__inference_B_pool4x2x2_1_layer_call_and_return_conditional_losses_630874¸_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Þ
.__inference_B_pool4x2x2_1_layer_call_fn_630880«_¢\
U¢R
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
!__inference__wrapped_model_630832&'45:;HIfgpqvw=¢:
3¢0
.+
input_1ÿÿÿÿÿÿÿÿÿÈ
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿÒ
G__inference_concatenate_layer_call_and_return_conditional_losses_632294\¢Y
R¢O
MJ
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ+
 ©
,__inference_concatenate_layer_call_fn_632300y\¢Y
R¢O
MJ
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ+¥
C__inference_dense_1_layer_call_and_return_conditional_losses_632385^pq0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_1_layer_call_fn_632394Qpq0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_2_layer_call_and_return_conditional_losses_632404]vw0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_2_layer_call_fn_632413Pvw0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_dense_layer_call_and_return_conditional_losses_632338^fg0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ+
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dense_layer_call_fn_632347Qfg0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ+
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_1_layer_call_and_return_conditional_losses_632359^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_1_layer_call_and_return_conditional_losses_632364^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_1_layer_call_fn_632369Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_1_layer_call_fn_632374Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_dropout_layer_call_and_return_conditional_losses_632312^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ+
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ+
 ¥
C__inference_dropout_layer_call_and_return_conditional_losses_632317^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ+
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ+
 }
(__inference_dropout_layer_call_fn_632322Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ+
p
ª "ÿÿÿÿÿÿÿÿÿ+}
(__inference_dropout_layer_call_fn_632327Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ+
p 
ª "ÿÿÿÿÿÿÿÿÿ+Í
H__inference_functional_3_layer_call_and_return_conditional_losses_631441&'45:;HIfgpqvwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Í
H__inference_functional_3_layer_call_and_return_conditional_losses_631497&'45:;HIfgpqvwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
H__inference_functional_3_layer_call_and_return_conditional_losses_631861&'45:;HIfgpqvwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
H__inference_functional_3_layer_call_and_return_conditional_losses_631935&'45:;HIfgpqvwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
-__inference_functional_3_layer_call_fn_631591s&'45:;HIfgpqvwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¤
-__inference_functional_3_layer_call_fn_631684s&'45:;HIfgpqvwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
-__inference_functional_3_layer_call_fn_631972r&'45:;HIfgpqvwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ£
-__inference_functional_3_layer_call_fn_632009r&'45:;HIfgpqvwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¸
$__inference_signature_wrapper_631735&'45:;HIfgpqvwH¢E
¢ 
>ª;
9
input_1.+
input_1ÿÿÿÿÿÿÿÿÿÈ"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿÇ
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632211t?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ 
 Ç
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632216t?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ 
 
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632250¼c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
O__inference_spatial_dropout3d_1_layer_call_and_return_conditional_losses_632255¼c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
4__inference_spatial_dropout3d_1_layer_call_fn_632221g?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "$!ÿÿÿÿÿÿÿÿÿ 
4__inference_spatial_dropout3d_1_layer_call_fn_632226g?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "$!ÿÿÿÿÿÿÿÿÿ è
4__inference_spatial_dropout3d_1_layer_call_fn_632260¯c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿè
4__inference_spatial_dropout3d_1_layer_call_fn_632265¯c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632113¼c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632118¼c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "U¢R
KH
0Aÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632152t?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ. 
p
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ. 
 Å
M__inference_spatial_dropout3d_layer_call_and_return_conditional_losses_632157t?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ. 
p 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ. 
 æ
2__inference_spatial_dropout3d_layer_call_fn_632123¯c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
2__inference_spatial_dropout3d_layer_call_fn_632128¯c¢`
Y¢V
PM
inputsAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "HEAÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2__inference_spatial_dropout3d_layer_call_fn_632162g?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ. 
p
ª "$!ÿÿÿÿÿÿÿÿÿ. 
2__inference_spatial_dropout3d_layer_call_fn_632167g?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ. 
p 
ª "$!ÿÿÿÿÿÿÿÿÿ. 