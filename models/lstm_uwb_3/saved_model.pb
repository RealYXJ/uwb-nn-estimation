��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02unknown8�
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
~
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/v/dense_3/kernel
�
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/m/dense_3/kernel
�
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
UAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*f
shared_nameWUAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias
�
iAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias/Read/ReadVariableOpReadVariableOpUAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias*
_output_shapes
:*
dtype0
�
UAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*f
shared_nameWUAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias
�
iAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias/Read/ReadVariableOpReadVariableOpUAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias*
_output_shapes
:*
dtype0
�
WAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*h
shared_nameYWAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel
�
kAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel/Read/ReadVariableOpReadVariableOpWAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel*"
_output_shapes
:*
dtype0
�
WAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*h
shared_nameYWAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel
�
kAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel/Read/ReadVariableOpReadVariableOpWAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel*"
_output_shapes
:*
dtype0
�
JAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/bias
�
^Adam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/bias/Read/ReadVariableOpReadVariableOpJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/bias*
_output_shapes

:*
dtype0
�
JAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/bias
�
^Adam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/bias/Read/ReadVariableOpReadVariableOpJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/bias*
_output_shapes

:*
dtype0
�
LAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel
�
`Adam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel/Read/ReadVariableOpReadVariableOpLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel*"
_output_shapes
:*
dtype0
�
LAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel
�
`Adam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel/Read/ReadVariableOpReadVariableOpLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel*"
_output_shapes
:*
dtype0
�
HAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Y
shared_nameJHAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/bias
�
\Adam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/bias/Read/ReadVariableOpReadVariableOpHAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/bias*
_output_shapes

:*
dtype0
�
HAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Y
shared_nameJHAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/bias
�
\Adam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/bias/Read/ReadVariableOpReadVariableOpHAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/bias*
_output_shapes

:*
dtype0
�
JAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel
�
^Adam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel/Read/ReadVariableOpReadVariableOpJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel*"
_output_shapes
:*
dtype0
�
JAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel
�
^Adam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel/Read/ReadVariableOpReadVariableOpJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel*"
_output_shapes
:*
dtype0
�
JAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/bias
�
^Adam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/bias/Read/ReadVariableOpReadVariableOpJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/bias*
_output_shapes

:*
dtype0
�
JAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/bias
�
^Adam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/bias/Read/ReadVariableOpReadVariableOpJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/bias*
_output_shapes

:*
dtype0
�
LAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel
�
`Adam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel/Read/ReadVariableOpReadVariableOpLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel*"
_output_shapes
:*
dtype0
�
LAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel
�
`Adam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel/Read/ReadVariableOpReadVariableOpLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel*"
_output_shapes
:*
dtype0
�
UAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*f
shared_nameWUAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias
�
iAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias/Read/ReadVariableOpReadVariableOpUAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias*
_output_shapes
:*
dtype0
�
UAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*f
shared_nameWUAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias
�
iAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias/Read/ReadVariableOpReadVariableOpUAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias*
_output_shapes
:*
dtype0
�
WAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*h
shared_nameYWAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel
�
kAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel/Read/ReadVariableOpReadVariableOpWAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel*"
_output_shapes
:*
dtype0
�
WAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*h
shared_nameYWAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel
�
kAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel/Read/ReadVariableOpReadVariableOpWAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel*"
_output_shapes
:*
dtype0
�
JAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/bias
�
^Adam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/bias/Read/ReadVariableOpReadVariableOpJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/bias*
_output_shapes

:*
dtype0
�
JAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/bias
�
^Adam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/bias/Read/ReadVariableOpReadVariableOpJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/bias*
_output_shapes

:*
dtype0
�
LAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel
�
`Adam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel/Read/ReadVariableOpReadVariableOpLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel*"
_output_shapes
:*
dtype0
�
LAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel
�
`Adam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel/Read/ReadVariableOpReadVariableOpLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel*"
_output_shapes
:*
dtype0
�
HAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Y
shared_nameJHAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/bias
�
\Adam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/bias/Read/ReadVariableOpReadVariableOpHAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/bias*
_output_shapes

:*
dtype0
�
HAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Y
shared_nameJHAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/bias
�
\Adam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/bias/Read/ReadVariableOpReadVariableOpHAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/bias*
_output_shapes

:*
dtype0
�
JAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel
�
^Adam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel/Read/ReadVariableOpReadVariableOpJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel*"
_output_shapes
:*
dtype0
�
JAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel
�
^Adam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel/Read/ReadVariableOpReadVariableOpJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel*"
_output_shapes
:*
dtype0
�
JAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/bias
�
^Adam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/bias/Read/ReadVariableOpReadVariableOpJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/bias*
_output_shapes

:*
dtype0
�
JAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/bias
�
^Adam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/bias/Read/ReadVariableOpReadVariableOpJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/bias*
_output_shapes

:*
dtype0
�
LAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel
�
`Adam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel/Read/ReadVariableOpReadVariableOpLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel*"
_output_shapes
:*
dtype0
�
LAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel
�
`Adam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel/Read/ReadVariableOpReadVariableOpLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel*"
_output_shapes
:*
dtype0
�
TAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*e
shared_nameVTAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias
�
hAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias/Read/ReadVariableOpReadVariableOpTAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias*
_output_shapes
:*
dtype0
�
TAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*e
shared_nameVTAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias
�
hAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias/Read/ReadVariableOpReadVariableOpTAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias*
_output_shapes
:*
dtype0
�
VAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*g
shared_nameXVAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel
�
jAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel/Read/ReadVariableOpReadVariableOpVAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel*"
_output_shapes
:*
dtype0
�
VAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*g
shared_nameXVAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel
�
jAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel/Read/ReadVariableOpReadVariableOpVAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel*"
_output_shapes
:*
dtype0
�
IAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Z
shared_nameKIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/bias
�
]Adam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/bias/Read/ReadVariableOpReadVariableOpIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/bias*
_output_shapes

:*
dtype0
�
IAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Z
shared_nameKIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/bias
�
]Adam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/bias/Read/ReadVariableOpReadVariableOpIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/bias*
_output_shapes

:*
dtype0
�
KAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*\
shared_nameMKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel
�
_Adam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel/Read/ReadVariableOpReadVariableOpKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel*"
_output_shapes
:*
dtype0
�
KAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*\
shared_nameMKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel
�
_Adam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel/Read/ReadVariableOpReadVariableOpKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel*"
_output_shapes
:*
dtype0
�
GAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*X
shared_nameIGAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/bias
�
[Adam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/bias/Read/ReadVariableOpReadVariableOpGAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/bias*
_output_shapes

:*
dtype0
�
GAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*X
shared_nameIGAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/bias
�
[Adam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/bias/Read/ReadVariableOpReadVariableOpGAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/bias*
_output_shapes

:*
dtype0
�
IAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel
�
]Adam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel/Read/ReadVariableOpReadVariableOpIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel*"
_output_shapes
:*
dtype0
�
IAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel
�
]Adam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel/Read/ReadVariableOpReadVariableOpIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel*"
_output_shapes
:*
dtype0
�
IAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Z
shared_nameKIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/bias
�
]Adam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/bias/Read/ReadVariableOpReadVariableOpIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/bias*
_output_shapes

:*
dtype0
�
IAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Z
shared_nameKIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/bias
�
]Adam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/bias/Read/ReadVariableOpReadVariableOpIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/bias*
_output_shapes

:*
dtype0
�
KAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*\
shared_nameMKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel
�
_Adam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel/Read/ReadVariableOpReadVariableOpKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel*"
_output_shapes
:*
dtype0
�
KAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*\
shared_nameMKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel
�
_Adam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel/Read/ReadVariableOpReadVariableOpKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel*"
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
Nmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*_
shared_namePNmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/bias
�
bmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/bias/Read/ReadVariableOpReadVariableOpNmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/bias*
_output_shapes
:*
dtype0
�
Pmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel
�
dmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel/Read/ReadVariableOpReadVariableOpPmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel*"
_output_shapes
:*
dtype0
�
Cmulti_scale_multi_head_attention/multi_head_attention_11/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*T
shared_nameECmulti_scale_multi_head_attention/multi_head_attention_11/value/bias
�
Wmulti_scale_multi_head_attention/multi_head_attention_11/value/bias/Read/ReadVariableOpReadVariableOpCmulti_scale_multi_head_attention/multi_head_attention_11/value/bias*
_output_shapes

:*
dtype0
�
Emulti_scale_multi_head_attention/multi_head_attention_11/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEmulti_scale_multi_head_attention/multi_head_attention_11/value/kernel
�
Ymulti_scale_multi_head_attention/multi_head_attention_11/value/kernel/Read/ReadVariableOpReadVariableOpEmulti_scale_multi_head_attention/multi_head_attention_11/value/kernel*"
_output_shapes
:*
dtype0
�
Amulti_scale_multi_head_attention/multi_head_attention_11/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAmulti_scale_multi_head_attention/multi_head_attention_11/key/bias
�
Umulti_scale_multi_head_attention/multi_head_attention_11/key/bias/Read/ReadVariableOpReadVariableOpAmulti_scale_multi_head_attention/multi_head_attention_11/key/bias*
_output_shapes

:*
dtype0
�
Cmulti_scale_multi_head_attention/multi_head_attention_11/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECmulti_scale_multi_head_attention/multi_head_attention_11/key/kernel
�
Wmulti_scale_multi_head_attention/multi_head_attention_11/key/kernel/Read/ReadVariableOpReadVariableOpCmulti_scale_multi_head_attention/multi_head_attention_11/key/kernel*"
_output_shapes
:*
dtype0
�
Cmulti_scale_multi_head_attention/multi_head_attention_11/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*T
shared_nameECmulti_scale_multi_head_attention/multi_head_attention_11/query/bias
�
Wmulti_scale_multi_head_attention/multi_head_attention_11/query/bias/Read/ReadVariableOpReadVariableOpCmulti_scale_multi_head_attention/multi_head_attention_11/query/bias*
_output_shapes

:*
dtype0
�
Emulti_scale_multi_head_attention/multi_head_attention_11/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEmulti_scale_multi_head_attention/multi_head_attention_11/query/kernel
�
Ymulti_scale_multi_head_attention/multi_head_attention_11/query/kernel/Read/ReadVariableOpReadVariableOpEmulti_scale_multi_head_attention/multi_head_attention_11/query/kernel*"
_output_shapes
:*
dtype0
�
Nmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*_
shared_namePNmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/bias
�
bmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/bias/Read/ReadVariableOpReadVariableOpNmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/bias*
_output_shapes
:*
dtype0
�
Pmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel
�
dmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel/Read/ReadVariableOpReadVariableOpPmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel*"
_output_shapes
:*
dtype0
�
Cmulti_scale_multi_head_attention/multi_head_attention_10/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*T
shared_nameECmulti_scale_multi_head_attention/multi_head_attention_10/value/bias
�
Wmulti_scale_multi_head_attention/multi_head_attention_10/value/bias/Read/ReadVariableOpReadVariableOpCmulti_scale_multi_head_attention/multi_head_attention_10/value/bias*
_output_shapes

:*
dtype0
�
Emulti_scale_multi_head_attention/multi_head_attention_10/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEmulti_scale_multi_head_attention/multi_head_attention_10/value/kernel
�
Ymulti_scale_multi_head_attention/multi_head_attention_10/value/kernel/Read/ReadVariableOpReadVariableOpEmulti_scale_multi_head_attention/multi_head_attention_10/value/kernel*"
_output_shapes
:*
dtype0
�
Amulti_scale_multi_head_attention/multi_head_attention_10/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAmulti_scale_multi_head_attention/multi_head_attention_10/key/bias
�
Umulti_scale_multi_head_attention/multi_head_attention_10/key/bias/Read/ReadVariableOpReadVariableOpAmulti_scale_multi_head_attention/multi_head_attention_10/key/bias*
_output_shapes

:*
dtype0
�
Cmulti_scale_multi_head_attention/multi_head_attention_10/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECmulti_scale_multi_head_attention/multi_head_attention_10/key/kernel
�
Wmulti_scale_multi_head_attention/multi_head_attention_10/key/kernel/Read/ReadVariableOpReadVariableOpCmulti_scale_multi_head_attention/multi_head_attention_10/key/kernel*"
_output_shapes
:*
dtype0
�
Cmulti_scale_multi_head_attention/multi_head_attention_10/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*T
shared_nameECmulti_scale_multi_head_attention/multi_head_attention_10/query/bias
�
Wmulti_scale_multi_head_attention/multi_head_attention_10/query/bias/Read/ReadVariableOpReadVariableOpCmulti_scale_multi_head_attention/multi_head_attention_10/query/bias*
_output_shapes

:*
dtype0
�
Emulti_scale_multi_head_attention/multi_head_attention_10/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEmulti_scale_multi_head_attention/multi_head_attention_10/query/kernel
�
Ymulti_scale_multi_head_attention/multi_head_attention_10/query/kernel/Read/ReadVariableOpReadVariableOpEmulti_scale_multi_head_attention/multi_head_attention_10/query/kernel*"
_output_shapes
:*
dtype0
�
Mmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*^
shared_nameOMmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/bias
�
amulti_scale_multi_head_attention/multi_head_attention_9/attention_output/bias/Read/ReadVariableOpReadVariableOpMmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/bias*
_output_shapes
:*
dtype0
�
Omulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*`
shared_nameQOmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel
�
cmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel/Read/ReadVariableOpReadVariableOpOmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel*"
_output_shapes
:*
dtype0
�
Bmulti_scale_multi_head_attention/multi_head_attention_9/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*S
shared_nameDBmulti_scale_multi_head_attention/multi_head_attention_9/value/bias
�
Vmulti_scale_multi_head_attention/multi_head_attention_9/value/bias/Read/ReadVariableOpReadVariableOpBmulti_scale_multi_head_attention/multi_head_attention_9/value/bias*
_output_shapes

:*
dtype0
�
Dmulti_scale_multi_head_attention/multi_head_attention_9/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDmulti_scale_multi_head_attention/multi_head_attention_9/value/kernel
�
Xmulti_scale_multi_head_attention/multi_head_attention_9/value/kernel/Read/ReadVariableOpReadVariableOpDmulti_scale_multi_head_attention/multi_head_attention_9/value/kernel*"
_output_shapes
:*
dtype0
�
@multi_scale_multi_head_attention/multi_head_attention_9/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Q
shared_nameB@multi_scale_multi_head_attention/multi_head_attention_9/key/bias
�
Tmulti_scale_multi_head_attention/multi_head_attention_9/key/bias/Read/ReadVariableOpReadVariableOp@multi_scale_multi_head_attention/multi_head_attention_9/key/bias*
_output_shapes

:*
dtype0
�
Bmulti_scale_multi_head_attention/multi_head_attention_9/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel
�
Vmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel/Read/ReadVariableOpReadVariableOpBmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel*"
_output_shapes
:*
dtype0
�
Bmulti_scale_multi_head_attention/multi_head_attention_9/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*S
shared_nameDBmulti_scale_multi_head_attention/multi_head_attention_9/query/bias
�
Vmulti_scale_multi_head_attention/multi_head_attention_9/query/bias/Read/ReadVariableOpReadVariableOpBmulti_scale_multi_head_attention/multi_head_attention_9/query/bias*
_output_shapes

:*
dtype0
�
Dmulti_scale_multi_head_attention/multi_head_attention_9/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDmulti_scale_multi_head_attention/multi_head_attention_9/query/kernel
�
Xmulti_scale_multi_head_attention/multi_head_attention_9/query/kernel/Read/ReadVariableOpReadVariableOpDmulti_scale_multi_head_attention/multi_head_attention_9/query/kernel*"
_output_shapes
:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_input_4Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4Dmulti_scale_multi_head_attention/multi_head_attention_9/query/kernelBmulti_scale_multi_head_attention/multi_head_attention_9/query/biasBmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel@multi_scale_multi_head_attention/multi_head_attention_9/key/biasDmulti_scale_multi_head_attention/multi_head_attention_9/value/kernelBmulti_scale_multi_head_attention/multi_head_attention_9/value/biasOmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelMmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/biasEmulti_scale_multi_head_attention/multi_head_attention_10/query/kernelCmulti_scale_multi_head_attention/multi_head_attention_10/query/biasCmulti_scale_multi_head_attention/multi_head_attention_10/key/kernelAmulti_scale_multi_head_attention/multi_head_attention_10/key/biasEmulti_scale_multi_head_attention/multi_head_attention_10/value/kernelCmulti_scale_multi_head_attention/multi_head_attention_10/value/biasPmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelNmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/biasEmulti_scale_multi_head_attention/multi_head_attention_11/query/kernelCmulti_scale_multi_head_attention/multi_head_attention_11/query/biasCmulti_scale_multi_head_attention/multi_head_attention_11/key/kernelAmulti_scale_multi_head_attention/multi_head_attention_11/key/biasEmulti_scale_multi_head_attention/multi_head_attention_11/value/kernelCmulti_scale_multi_head_attention/multi_head_attention_11/value/biasPmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelNmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/biasdense_3/kerneldense_3/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_60907

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
attention_layers*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias*
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
 24
!25*
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
 24
!25*
* 
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
?trace_0
@trace_1
Atrace_2
Btrace_3* 
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
* 
�
G
_variables
H_iterations
I_learning_rate
J_index_dict
K
_momentums
L_velocities
M_update_step_xla*

Nserving_default* 
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923*
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923*
* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ttrace_0
Utrace_1* 

Vtrace_0
Wtrace_1* 

X0
Y1
Z2*
* 
* 
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

`trace_0* 

atrace_0* 

 0
!1*

 0
!1*
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEDmulti_scale_multi_head_attention/multi_head_attention_9/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEBmulti_scale_multi_head_attention/multi_head_attention_9/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEBmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE@multi_scale_multi_head_attention/multi_head_attention_9/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEDmulti_scale_multi_head_attention/multi_head_attention_9/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEBmulti_scale_multi_head_attention/multi_head_attention_9/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEOmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEMmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEEmulti_scale_multi_head_attention/multi_head_attention_10/query/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUECmulti_scale_multi_head_attention/multi_head_attention_10/query/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUECmulti_scale_multi_head_attention/multi_head_attention_10/key/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAmulti_scale_multi_head_attention/multi_head_attention_10/key/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEEmulti_scale_multi_head_attention/multi_head_attention_10/value/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUECmulti_scale_multi_head_attention/multi_head_attention_10/value/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEPmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUENmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEEmulti_scale_multi_head_attention/multi_head_attention_11/query/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUECmulti_scale_multi_head_attention/multi_head_attention_11/query/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUECmulti_scale_multi_head_attention/multi_head_attention_11/key/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAmulti_scale_multi_head_attention/multi_head_attention_11/key/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEEmulti_scale_multi_head_attention/multi_head_attention_11/value/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUECmulti_scale_multi_head_attention/multi_head_attention_11/value/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEPmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUENmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

i0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
H0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15
y16
z17
{18
|19
}20
~21
22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
j0
l1
n2
p3
r4
t5
v6
x7
z8
|9
~10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25*
�
k0
m1
o2
q3
s4
u5
w6
y7
{8
}9
10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25*
* 
* 
* 

X0
Y1
Z2*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
��
VARIABLE_VALUEKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEGAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEGAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEVAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEVAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUETAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUETAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEWAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEWAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUELAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEWAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEWAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
<
"0
#1
$2
%3
&4
'5
(6
)7*
<
"0
#1
$2
%3
&4
'5
(6
)7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

"kernel
#bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

$kernel
%bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

&kernel
'bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

(kernel
)bias*
<
*0
+1
,2
-3
.4
/5
06
17*
<
*0
+1
,2
-3
.4
/5
06
17*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

*kernel
+bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

,kernel
-bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

.kernel
/bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

0kernel
1bias*
<
20
31
42
53
64
75
86
97*
<
20
31
42
53
64
75
86
97*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

2kernel
3bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

4kernel
5bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

6kernel
7bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

8kernel
9bias*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

$0
%1*

$0
%1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

&0
'1*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

(0
)1*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

*0
+1*

*0
+1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

,0
-1*

,0
-1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

.0
/1*

.0
/1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

00
11*

00
11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

20
31*

20
31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

40
51*

40
51*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

80
91*

80
91*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasDmulti_scale_multi_head_attention/multi_head_attention_9/query/kernelBmulti_scale_multi_head_attention/multi_head_attention_9/query/biasBmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel@multi_scale_multi_head_attention/multi_head_attention_9/key/biasDmulti_scale_multi_head_attention/multi_head_attention_9/value/kernelBmulti_scale_multi_head_attention/multi_head_attention_9/value/biasOmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelMmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/biasEmulti_scale_multi_head_attention/multi_head_attention_10/query/kernelCmulti_scale_multi_head_attention/multi_head_attention_10/query/biasCmulti_scale_multi_head_attention/multi_head_attention_10/key/kernelAmulti_scale_multi_head_attention/multi_head_attention_10/key/biasEmulti_scale_multi_head_attention/multi_head_attention_10/value/kernelCmulti_scale_multi_head_attention/multi_head_attention_10/value/biasPmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelNmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/biasEmulti_scale_multi_head_attention/multi_head_attention_11/query/kernelCmulti_scale_multi_head_attention/multi_head_attention_11/query/biasCmulti_scale_multi_head_attention/multi_head_attention_11/key/kernelAmulti_scale_multi_head_attention/multi_head_attention_11/key/biasEmulti_scale_multi_head_attention/multi_head_attention_11/value/kernelCmulti_scale_multi_head_attention/multi_head_attention_11/value/biasPmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelNmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/bias	iterationlearning_rateKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernelKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernelIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/biasIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/biasIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernelIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernelGAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/biasGAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/biasKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernelKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernelIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/biasIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/biasVAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelVAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelTAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/biasTAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/biasJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernelJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernelHAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/biasHAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/biasWAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelWAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelUAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/biasUAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/biasJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernelJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernelHAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/biasHAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/biasWAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelWAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelUAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/biasUAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotalcountConst*_
TinX
V2T*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_62072
�/
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasDmulti_scale_multi_head_attention/multi_head_attention_9/query/kernelBmulti_scale_multi_head_attention/multi_head_attention_9/query/biasBmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel@multi_scale_multi_head_attention/multi_head_attention_9/key/biasDmulti_scale_multi_head_attention/multi_head_attention_9/value/kernelBmulti_scale_multi_head_attention/multi_head_attention_9/value/biasOmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelMmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/biasEmulti_scale_multi_head_attention/multi_head_attention_10/query/kernelCmulti_scale_multi_head_attention/multi_head_attention_10/query/biasCmulti_scale_multi_head_attention/multi_head_attention_10/key/kernelAmulti_scale_multi_head_attention/multi_head_attention_10/key/biasEmulti_scale_multi_head_attention/multi_head_attention_10/value/kernelCmulti_scale_multi_head_attention/multi_head_attention_10/value/biasPmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelNmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/biasEmulti_scale_multi_head_attention/multi_head_attention_11/query/kernelCmulti_scale_multi_head_attention/multi_head_attention_11/query/biasCmulti_scale_multi_head_attention/multi_head_attention_11/key/kernelAmulti_scale_multi_head_attention/multi_head_attention_11/key/biasEmulti_scale_multi_head_attention/multi_head_attention_11/value/kernelCmulti_scale_multi_head_attention/multi_head_attention_11/value/biasPmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelNmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/bias	iterationlearning_rateKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernelKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernelIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/biasIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/biasIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernelIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernelGAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/biasGAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/biasKAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernelKAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernelIAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/biasIAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/biasVAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelVAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernelTAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/biasTAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/biasJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernelJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernelHAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/biasHAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/biasWAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelWAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernelUAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/biasUAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/biasJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernelJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernelHAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/biasHAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/biasLAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernelLAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernelJAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/biasJAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/biasWAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelWAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernelUAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/biasUAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotalcount*^
TinW
U2S*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_62328��
�#
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60376
input_4<
&multi_scale_multi_head_attention_60320:8
&multi_scale_multi_head_attention_60322:<
&multi_scale_multi_head_attention_60324:8
&multi_scale_multi_head_attention_60326:<
&multi_scale_multi_head_attention_60328:8
&multi_scale_multi_head_attention_60330:<
&multi_scale_multi_head_attention_60332:4
&multi_scale_multi_head_attention_60334:<
&multi_scale_multi_head_attention_60336:8
&multi_scale_multi_head_attention_60338:<
&multi_scale_multi_head_attention_60340:8
&multi_scale_multi_head_attention_60342:<
&multi_scale_multi_head_attention_60344:8
&multi_scale_multi_head_attention_60346:<
&multi_scale_multi_head_attention_60348:4
&multi_scale_multi_head_attention_60350:<
&multi_scale_multi_head_attention_60352:8
&multi_scale_multi_head_attention_60354:<
&multi_scale_multi_head_attention_60356:8
&multi_scale_multi_head_attention_60358:<
&multi_scale_multi_head_attention_60360:8
&multi_scale_multi_head_attention_60362:<
&multi_scale_multi_head_attention_60364:4
&multi_scale_multi_head_attention_60366: 
dense_3_60370:	�
dense_3_60372:
identity��dense_3/StatefulPartitionedCall�8multi_scale_multi_head_attention/StatefulPartitionedCall�

8multi_scale_multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinput_4&multi_scale_multi_head_attention_60320&multi_scale_multi_head_attention_60322&multi_scale_multi_head_attention_60324&multi_scale_multi_head_attention_60326&multi_scale_multi_head_attention_60328&multi_scale_multi_head_attention_60330&multi_scale_multi_head_attention_60332&multi_scale_multi_head_attention_60334&multi_scale_multi_head_attention_60336&multi_scale_multi_head_attention_60338&multi_scale_multi_head_attention_60340&multi_scale_multi_head_attention_60342&multi_scale_multi_head_attention_60344&multi_scale_multi_head_attention_60346&multi_scale_multi_head_attention_60348&multi_scale_multi_head_attention_60350&multi_scale_multi_head_attention_60352&multi_scale_multi_head_attention_60354&multi_scale_multi_head_attention_60356&multi_scale_multi_head_attention_60358&multi_scale_multi_head_attention_60360&multi_scale_multi_head_attention_60362&multi_scale_multi_head_attention_60364&multi_scale_multi_head_attention_60366*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60319�
flatten_3/PartitionedCallPartitionedCallAmulti_scale_multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_60202�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_60370dense_3_60372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_60214w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_3/StatefulPartitionedCall9^multi_scale_multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2t
8multi_scale_multi_head_attention/StatefulPartitionedCall8multi_scale_multi_head_attention/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_4
�#
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60438

inputs<
&multi_scale_multi_head_attention_60382:8
&multi_scale_multi_head_attention_60384:<
&multi_scale_multi_head_attention_60386:8
&multi_scale_multi_head_attention_60388:<
&multi_scale_multi_head_attention_60390:8
&multi_scale_multi_head_attention_60392:<
&multi_scale_multi_head_attention_60394:4
&multi_scale_multi_head_attention_60396:<
&multi_scale_multi_head_attention_60398:8
&multi_scale_multi_head_attention_60400:<
&multi_scale_multi_head_attention_60402:8
&multi_scale_multi_head_attention_60404:<
&multi_scale_multi_head_attention_60406:8
&multi_scale_multi_head_attention_60408:<
&multi_scale_multi_head_attention_60410:4
&multi_scale_multi_head_attention_60412:<
&multi_scale_multi_head_attention_60414:8
&multi_scale_multi_head_attention_60416:<
&multi_scale_multi_head_attention_60418:8
&multi_scale_multi_head_attention_60420:<
&multi_scale_multi_head_attention_60422:8
&multi_scale_multi_head_attention_60424:<
&multi_scale_multi_head_attention_60426:4
&multi_scale_multi_head_attention_60428: 
dense_3_60432:	�
dense_3_60434:
identity��dense_3/StatefulPartitionedCall�8multi_scale_multi_head_attention/StatefulPartitionedCall�

8multi_scale_multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinputs&multi_scale_multi_head_attention_60382&multi_scale_multi_head_attention_60384&multi_scale_multi_head_attention_60386&multi_scale_multi_head_attention_60388&multi_scale_multi_head_attention_60390&multi_scale_multi_head_attention_60392&multi_scale_multi_head_attention_60394&multi_scale_multi_head_attention_60396&multi_scale_multi_head_attention_60398&multi_scale_multi_head_attention_60400&multi_scale_multi_head_attention_60402&multi_scale_multi_head_attention_60404&multi_scale_multi_head_attention_60406&multi_scale_multi_head_attention_60408&multi_scale_multi_head_attention_60410&multi_scale_multi_head_attention_60412&multi_scale_multi_head_attention_60414&multi_scale_multi_head_attention_60416&multi_scale_multi_head_attention_60418&multi_scale_multi_head_attention_60420&multi_scale_multi_head_attention_60422&multi_scale_multi_head_attention_60424&multi_scale_multi_head_attention_60426&multi_scale_multi_head_attention_60428*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60146�
flatten_3/PartitionedCallPartitionedCallAmulti_scale_multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_60202�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_60432dense_3_60434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_60214w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_3/StatefulPartitionedCall9^multi_scale_multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2t
8multi_scale_multi_head_attention/StatefulPartitionedCall8multi_scale_multi_head_attention/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_sequential_3_layer_call_fn_60964

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_60438o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�)
G__inference_sequential_3_layer_call_and_return_conditional_losses_61229

inputsy
cmulti_scale_multi_head_attention_multi_head_attention_9_query_einsum_einsum_readvariableop_resource:k
Ymulti_scale_multi_head_attention_multi_head_attention_9_query_add_readvariableop_resource:w
amulti_scale_multi_head_attention_multi_head_attention_9_key_einsum_einsum_readvariableop_resource:i
Wmulti_scale_multi_head_attention_multi_head_attention_9_key_add_readvariableop_resource:y
cmulti_scale_multi_head_attention_multi_head_attention_9_value_einsum_einsum_readvariableop_resource:k
Ymulti_scale_multi_head_attention_multi_head_attention_9_value_add_readvariableop_resource:�
nmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource:r
dmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_10_query_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_10_query_add_readvariableop_resource:x
bmulti_scale_multi_head_attention_multi_head_attention_10_key_einsum_einsum_readvariableop_resource:j
Xmulti_scale_multi_head_attention_multi_head_attention_10_key_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_10_value_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_10_value_add_readvariableop_resource:�
omulti_scale_multi_head_attention_multi_head_attention_10_attention_output_einsum_einsum_readvariableop_resource:s
emulti_scale_multi_head_attention_multi_head_attention_10_attention_output_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_11_query_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_11_query_add_readvariableop_resource:x
bmulti_scale_multi_head_attention_multi_head_attention_11_key_einsum_einsum_readvariableop_resource:j
Xmulti_scale_multi_head_attention_multi_head_attention_11_key_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_11_value_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_11_value_add_readvariableop_resource:�
omulti_scale_multi_head_attention_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource:s
emulti_scale_multi_head_attention_multi_head_attention_11_attention_output_add_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	�5
'dense_3_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp�fmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp�Omulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp�Ymulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp�\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp�fmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp�Omulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp�Ymulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp�emulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp�Nmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp�Xmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp�Pmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp�Zmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp�Pmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp�Zmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp�
Zmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpcmulti_scale_multi_head_attention_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Kmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/EinsumEinsuminputsbmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Pmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpYmulti_scale_multi_head_attention_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Amulti_scale_multi_head_attention/multi_head_attention_9/query/addAddV2Tmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum:output:0Xmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Xmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOpamulti_scale_multi_head_attention_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Imulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/EinsumEinsuminputs`multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Nmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpWmulti_scale_multi_head_attention_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
?multi_scale_multi_head_attention/multi_head_attention_9/key/addAddV2Rmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum:output:0Vmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Zmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpcmulti_scale_multi_head_attention_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Kmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/EinsumEinsuminputsbmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Pmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpYmulti_scale_multi_head_attention_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Amulti_scale_multi_head_attention/multi_head_attention_9/value/addAddV2Tmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum:output:0Xmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
=multi_scale_multi_head_attention/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
;multi_scale_multi_head_attention/multi_head_attention_9/MulMulEmulti_scale_multi_head_attention/multi_head_attention_9/query/add:z:0Fmulti_scale_multi_head_attention/multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:����������
Emulti_scale_multi_head_attention/multi_head_attention_9/einsum/EinsumEinsumCmulti_scale_multi_head_attention/multi_head_attention_9/key/add:z:0?multi_scale_multi_head_attention/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Gmulti_scale_multi_head_attention/multi_head_attention_9/softmax/SoftmaxSoftmaxNmulti_scale_multi_head_attention/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Hmulti_scale_multi_head_attention/multi_head_attention_9/dropout/IdentityIdentityQmulti_scale_multi_head_attention/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Gmulti_scale_multi_head_attention/multi_head_attention_9/einsum_1/EinsumEinsumQmulti_scale_multi_head_attention/multi_head_attention_9/dropout/Identity:output:0Emulti_scale_multi_head_attention/multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
emulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpnmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Vmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/EinsumEinsumPmulti_scale_multi_head_attention/multi_head_attention_9/einsum_1/Einsum:output:0mmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/addAddV2_multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum:output:0cmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_10_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_10/query/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_10_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_10/query/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Ymulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpReadVariableOpbmulti_scale_multi_head_attention_multi_head_attention_10_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Jmulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/EinsumEinsuminputsamulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Omulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpReadVariableOpXmulti_scale_multi_head_attention_multi_head_attention_10_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
@multi_scale_multi_head_attention/multi_head_attention_10/key/addAddV2Smulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum:output:0Wmulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_10_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_10/value/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_10_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_10/value/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
>multi_scale_multi_head_attention/multi_head_attention_10/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
<multi_scale_multi_head_attention/multi_head_attention_10/MulMulFmulti_scale_multi_head_attention/multi_head_attention_10/query/add:z:0Gmulti_scale_multi_head_attention/multi_head_attention_10/Mul/y:output:0*
T0*/
_output_shapes
:����������
Fmulti_scale_multi_head_attention/multi_head_attention_10/einsum/EinsumEinsumDmulti_scale_multi_head_attention/multi_head_attention_10/key/add:z:0@multi_scale_multi_head_attention/multi_head_attention_10/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Jmulti_scale_multi_head_attention/multi_head_attention_10/softmax_1/SoftmaxSoftmaxOmulti_scale_multi_head_attention/multi_head_attention_10/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Kmulti_scale_multi_head_attention/multi_head_attention_10/dropout_1/IdentityIdentityTmulti_scale_multi_head_attention/multi_head_attention_10/softmax_1/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Hmulti_scale_multi_head_attention/multi_head_attention_10/einsum_1/EinsumEinsumTmulti_scale_multi_head_attention/multi_head_attention_10/dropout_1/Identity:output:0Fmulti_scale_multi_head_attention/multi_head_attention_10/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
fmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpomulti_scale_multi_head_attention_multi_head_attention_10_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Wmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/EinsumEinsumQmulti_scale_multi_head_attention/multi_head_attention_10/einsum_1/Einsum:output:0nmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOpReadVariableOpemulti_scale_multi_head_attention_multi_head_attention_10_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Mmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/addAddV2`multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum:output:0dmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_11/query/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_11/query/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Ymulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpbmulti_scale_multi_head_attention_multi_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Jmulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/EinsumEinsuminputsamulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Omulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpReadVariableOpXmulti_scale_multi_head_attention_multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
@multi_scale_multi_head_attention/multi_head_attention_11/key/addAddV2Smulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum:output:0Wmulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_11/value/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_11/value/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
>multi_scale_multi_head_attention/multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
<multi_scale_multi_head_attention/multi_head_attention_11/MulMulFmulti_scale_multi_head_attention/multi_head_attention_11/query/add:z:0Gmulti_scale_multi_head_attention/multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:����������
Fmulti_scale_multi_head_attention/multi_head_attention_11/einsum/EinsumEinsumDmulti_scale_multi_head_attention/multi_head_attention_11/key/add:z:0@multi_scale_multi_head_attention/multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Jmulti_scale_multi_head_attention/multi_head_attention_11/softmax_2/SoftmaxSoftmaxOmulti_scale_multi_head_attention/multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Kmulti_scale_multi_head_attention/multi_head_attention_11/dropout_2/IdentityIdentityTmulti_scale_multi_head_attention/multi_head_attention_11/softmax_2/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Hmulti_scale_multi_head_attention/multi_head_attention_11/einsum_1/EinsumEinsumTmulti_scale_multi_head_attention/multi_head_attention_11/dropout_2/Identity:output:0Fmulti_scale_multi_head_attention/multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
fmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpomulti_scale_multi_head_attention_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Wmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/EinsumEinsumQmulti_scale_multi_head_attention/multi_head_attention_11/einsum_1/Einsum:output:0nmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpemulti_scale_multi_head_attention_multi_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Mmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/addAddV2`multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum:output:0dmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w
,multi_scale_multi_head_attention/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
'multi_scale_multi_head_attention/concatConcatV2Pmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/add:z:0Qmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/add:z:0Qmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/add:z:05multi_scale_multi_head_attention/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  �
flatten_3/ReshapeReshape0multi_scale_multi_head_attention/concat:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp]^multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOpg^multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpP^multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpZ^multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp]^multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOpg^multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpP^multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpZ^multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOpf^multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpO^multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpY^multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpQ^multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp[^multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpQ^multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp[^multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2�
\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp2�
fmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpfmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp2�
Omulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpOmulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp2�
Ymulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpYmulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp2�
\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp2�
fmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpfmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2�
Omulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpOmulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp2�
Ymulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpYmulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp2�
emulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpemulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2�
Nmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpNmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp2�
Xmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpXmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2�
Pmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOpPmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp2�
Zmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpZmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2�
Pmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOpPmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp2�
Zmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpZmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
ɣ
�k
__inference__traced_save_62072
file_prefix8
%read_disablecopyonread_dense_3_kernel:	�3
%read_1_disablecopyonread_dense_3_bias:s
]read_2_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel:m
[read_3_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_query_bias:q
[read_4_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel:k
Yread_5_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_key_bias:s
]read_6_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel:m
[read_7_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_value_bias:~
hread_8_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel:t
fread_9_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias:u
_read_10_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel:o
]read_11_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_query_bias:s
]read_12_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel:m
[read_13_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_key_bias:u
_read_14_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel:o
]read_15_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_value_bias:�
jread_16_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel:v
hread_17_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias:u
_read_18_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel:o
]read_19_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_query_bias:s
]read_20_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel:m
[read_21_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_key_bias:u
_read_22_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel:o
]read_23_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_value_bias:�
jread_24_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel:v
hread_25_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias:-
#read_26_disablecopyonread_iteration:	 1
'read_27_disablecopyonread_learning_rate: {
eread_28_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel:{
eread_29_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel:u
cread_30_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_bias:u
cread_31_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_bias:y
cread_32_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel:y
cread_33_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel:s
aread_34_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_bias:s
aread_35_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_bias:{
eread_36_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel:{
eread_37_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel:u
cread_38_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_bias:u
cread_39_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_bias:�
pread_40_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel:�
pread_41_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel:|
nread_42_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias:|
nread_43_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias:|
fread_44_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel:|
fread_45_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel:v
dread_46_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_bias:v
dread_47_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_bias:z
dread_48_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel:z
dread_49_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel:t
bread_50_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_bias:t
bread_51_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_bias:|
fread_52_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel:|
fread_53_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel:v
dread_54_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_bias:v
dread_55_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_bias:�
qread_56_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel:�
qread_57_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel:}
oread_58_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias:}
oread_59_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias:|
fread_60_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel:|
fread_61_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel:v
dread_62_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_bias:v
dread_63_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_bias:z
dread_64_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel:z
dread_65_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel:t
bread_66_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_bias:t
bread_67_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_bias:|
fread_68_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel:|
fread_69_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel:v
dread_70_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_bias:v
dread_71_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_bias:�
qread_72_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel:�
qread_73_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel:}
oread_74_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias:}
oread_75_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias:B
/read_76_disablecopyonread_adam_m_dense_3_kernel:	�B
/read_77_disablecopyonread_adam_v_dense_3_kernel:	�;
-read_78_disablecopyonread_adam_m_dense_3_bias:;
-read_79_disablecopyonread_adam_v_dense_3_bias:)
read_80_disablecopyonread_total: )
read_81_disablecopyonread_count: 
savev2_const
identity_165��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_3_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_3_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead]read_2_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp]read_2_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0q

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:g

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead[read_3_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_query_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp[read_3_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_query_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_4/DisableCopyOnReadDisableCopyOnRead[read_4_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp[read_4_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0q

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:g

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnReadYread_5_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_key_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpYread_5_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_key_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_6/DisableCopyOnReadDisableCopyOnRead]read_6_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp]read_6_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_7/DisableCopyOnReadDisableCopyOnRead[read_7_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_value_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp[read_7_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_value_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_8/DisableCopyOnReadDisableCopyOnReadhread_8_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOphread_8_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0r
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnReadfread_9_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpfread_9_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead_read_10_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp_read_10_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead]read_11_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_query_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp]read_11_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_query_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_12/DisableCopyOnReadDisableCopyOnRead]read_12_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp]read_12_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnRead[read_13_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_key_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp[read_13_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_key_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_14/DisableCopyOnReadDisableCopyOnRead_read_14_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp_read_14_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead]read_15_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_value_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp]read_15_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_value_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_16/DisableCopyOnReadDisableCopyOnReadjread_16_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpjread_16_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnReadhread_17_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOphread_17_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead_read_18_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp_read_18_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead]read_19_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_query_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp]read_19_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_query_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_20/DisableCopyOnReadDisableCopyOnRead]read_20_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp]read_20_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead[read_21_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_key_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp[read_21_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_key_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_22/DisableCopyOnReadDisableCopyOnRead_read_22_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp_read_22_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead]read_23_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_value_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp]read_23_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_value_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_24/DisableCopyOnReadDisableCopyOnReadjread_24_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpjread_24_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnReadhread_25_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOphread_25_disablecopyonread_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_26/DisableCopyOnReadDisableCopyOnRead#read_26_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp#read_26_disablecopyonread_iteration^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_27/DisableCopyOnReadDisableCopyOnRead'read_27_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp'read_27_disablecopyonread_learning_rate^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnReaderead_28_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOperead_28_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_29/DisableCopyOnReadDisableCopyOnReaderead_29_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOperead_29_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_30/DisableCopyOnReadDisableCopyOnReadcread_30_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpcread_30_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_31/DisableCopyOnReadDisableCopyOnReadcread_31_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpcread_31_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_32/DisableCopyOnReadDisableCopyOnReadcread_32_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpcread_32_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnReadcread_33_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpcread_33_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnReadaread_34_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOparead_34_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_35/DisableCopyOnReadDisableCopyOnReadaread_35_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOparead_35_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_36/DisableCopyOnReadDisableCopyOnReaderead_36_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOperead_36_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnReaderead_37_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOperead_37_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_38/DisableCopyOnReadDisableCopyOnReadcread_38_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpcread_38_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_39/DisableCopyOnReadDisableCopyOnReadcread_39_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpcread_39_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_40/DisableCopyOnReadDisableCopyOnReadpread_40_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOppread_40_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_41/DisableCopyOnReadDisableCopyOnReadpread_41_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOppread_41_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_42/DisableCopyOnReadDisableCopyOnReadnread_42_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpnread_42_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnReadnread_43_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpnread_43_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnReadfread_44_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpfread_44_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnReadfread_45_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpfread_45_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnReaddread_46_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpdread_46_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_47/DisableCopyOnReadDisableCopyOnReaddread_47_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpdread_47_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_48/DisableCopyOnReadDisableCopyOnReaddread_48_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOpdread_48_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_49/DisableCopyOnReadDisableCopyOnReaddread_49_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpdread_49_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_50/DisableCopyOnReadDisableCopyOnReadbread_50_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpbread_50_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_51/DisableCopyOnReadDisableCopyOnReadbread_51_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpbread_51_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_52/DisableCopyOnReadDisableCopyOnReadfread_52_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpfread_52_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_53/DisableCopyOnReadDisableCopyOnReadfread_53_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpfread_53_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_54/DisableCopyOnReadDisableCopyOnReaddread_54_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpdread_54_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_55/DisableCopyOnReadDisableCopyOnReaddread_55_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpdread_55_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_56/DisableCopyOnReadDisableCopyOnReadqread_56_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpqread_56_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_57/DisableCopyOnReadDisableCopyOnReadqread_57_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpqread_57_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_58/DisableCopyOnReadDisableCopyOnReadoread_58_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOporead_58_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_59/DisableCopyOnReadDisableCopyOnReadoread_59_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOporead_59_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_60/DisableCopyOnReadDisableCopyOnReadfread_60_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOpfread_60_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel^Read_60/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_61/DisableCopyOnReadDisableCopyOnReadfread_61_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOpfread_61_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel^Read_61/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_62/DisableCopyOnReadDisableCopyOnReaddread_62_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_bias"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOpdread_62_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_bias^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_63/DisableCopyOnReadDisableCopyOnReaddread_63_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_bias"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOpdread_63_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_bias^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_64/DisableCopyOnReadDisableCopyOnReaddread_64_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOpdread_64_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_65/DisableCopyOnReadDisableCopyOnReaddread_65_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOpdread_65_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_66/DisableCopyOnReadDisableCopyOnReadbread_66_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_bias"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOpbread_66_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_67/DisableCopyOnReadDisableCopyOnReadbread_67_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOpbread_67_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_68/DisableCopyOnReadDisableCopyOnReadfread_68_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOpfread_68_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel^Read_68/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_69/DisableCopyOnReadDisableCopyOnReadfread_69_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOpfread_69_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel^Read_69/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_70/DisableCopyOnReadDisableCopyOnReaddread_70_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_bias"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOpdread_70_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_bias^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_71/DisableCopyOnReadDisableCopyOnReaddread_71_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_bias"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOpdread_71_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_bias^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_72/DisableCopyOnReadDisableCopyOnReadqread_72_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOpqread_72_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_73/DisableCopyOnReadDisableCopyOnReadqread_73_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOpqread_73_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_74/DisableCopyOnReadDisableCopyOnReadoread_74_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOporead_74_disablecopyonread_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_75/DisableCopyOnReadDisableCopyOnReadoread_75_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOporead_75_disablecopyonread_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_76/DisableCopyOnReadDisableCopyOnRead/read_76_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp/read_76_disablecopyonread_adam_m_dense_3_kernel^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_77/DisableCopyOnReadDisableCopyOnRead/read_77_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp/read_77_disablecopyonread_adam_v_dense_3_kernel^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_78/DisableCopyOnReadDisableCopyOnRead-read_78_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp-read_78_disablecopyonread_adam_m_dense_3_bias^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_79/DisableCopyOnReadDisableCopyOnRead-read_79_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp-read_79_disablecopyonread_adam_v_dense_3_bias^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_80/DisableCopyOnReadDisableCopyOnReadread_80_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOpread_80_disablecopyonread_total^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_81/DisableCopyOnReadDisableCopyOnReadread_81_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOpread_81_disablecopyonread_count^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
: � 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�
value�B�SB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�
value�B�SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *a
dtypesW
U2S	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_164Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_165IdentityIdentity_164:output:0^NoOp*
T0*
_output_shapes
: �"
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_165Identity_165:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:S

_output_shapes
: 
�
�
,__inference_sequential_3_layer_call_fn_61021

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_60554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_sequential_3_layer_call_fn_60609
input_4
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_60554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_4
�#
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60221
input_4<
&multi_scale_multi_head_attention_60147:8
&multi_scale_multi_head_attention_60149:<
&multi_scale_multi_head_attention_60151:8
&multi_scale_multi_head_attention_60153:<
&multi_scale_multi_head_attention_60155:8
&multi_scale_multi_head_attention_60157:<
&multi_scale_multi_head_attention_60159:4
&multi_scale_multi_head_attention_60161:<
&multi_scale_multi_head_attention_60163:8
&multi_scale_multi_head_attention_60165:<
&multi_scale_multi_head_attention_60167:8
&multi_scale_multi_head_attention_60169:<
&multi_scale_multi_head_attention_60171:8
&multi_scale_multi_head_attention_60173:<
&multi_scale_multi_head_attention_60175:4
&multi_scale_multi_head_attention_60177:<
&multi_scale_multi_head_attention_60179:8
&multi_scale_multi_head_attention_60181:<
&multi_scale_multi_head_attention_60183:8
&multi_scale_multi_head_attention_60185:<
&multi_scale_multi_head_attention_60187:8
&multi_scale_multi_head_attention_60189:<
&multi_scale_multi_head_attention_60191:4
&multi_scale_multi_head_attention_60193: 
dense_3_60215:	�
dense_3_60217:
identity��dense_3/StatefulPartitionedCall�8multi_scale_multi_head_attention/StatefulPartitionedCall�

8multi_scale_multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinput_4&multi_scale_multi_head_attention_60147&multi_scale_multi_head_attention_60149&multi_scale_multi_head_attention_60151&multi_scale_multi_head_attention_60153&multi_scale_multi_head_attention_60155&multi_scale_multi_head_attention_60157&multi_scale_multi_head_attention_60159&multi_scale_multi_head_attention_60161&multi_scale_multi_head_attention_60163&multi_scale_multi_head_attention_60165&multi_scale_multi_head_attention_60167&multi_scale_multi_head_attention_60169&multi_scale_multi_head_attention_60171&multi_scale_multi_head_attention_60173&multi_scale_multi_head_attention_60175&multi_scale_multi_head_attention_60177&multi_scale_multi_head_attention_60179&multi_scale_multi_head_attention_60181&multi_scale_multi_head_attention_60183&multi_scale_multi_head_attention_60185&multi_scale_multi_head_attention_60187&multi_scale_multi_head_attention_60189&multi_scale_multi_head_attention_60191&multi_scale_multi_head_attention_60193*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60146�
flatten_3/PartitionedCallPartitionedCallAmulti_scale_multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_60202�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_60215dense_3_60217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_60214w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_3/StatefulPartitionedCall9^multi_scale_multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2t
8multi_scale_multi_head_attention/StatefulPartitionedCall8multi_scale_multi_head_attention/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_4
��
�S
!__inference__traced_restore_62328
file_prefix2
assignvariableop_dense_3_kernel:	�-
assignvariableop_1_dense_3_bias:m
Wassignvariableop_2_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel:g
Uassignvariableop_3_multi_scale_multi_head_attention_multi_head_attention_9_query_bias:k
Uassignvariableop_4_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel:e
Sassignvariableop_5_multi_scale_multi_head_attention_multi_head_attention_9_key_bias:m
Wassignvariableop_6_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel:g
Uassignvariableop_7_multi_scale_multi_head_attention_multi_head_attention_9_value_bias:x
bassignvariableop_8_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel:n
`assignvariableop_9_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias:o
Yassignvariableop_10_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel:i
Wassignvariableop_11_multi_scale_multi_head_attention_multi_head_attention_10_query_bias:m
Wassignvariableop_12_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel:g
Uassignvariableop_13_multi_scale_multi_head_attention_multi_head_attention_10_key_bias:o
Yassignvariableop_14_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel:i
Wassignvariableop_15_multi_scale_multi_head_attention_multi_head_attention_10_value_bias:z
dassignvariableop_16_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel:p
bassignvariableop_17_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias:o
Yassignvariableop_18_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel:i
Wassignvariableop_19_multi_scale_multi_head_attention_multi_head_attention_11_query_bias:m
Wassignvariableop_20_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel:g
Uassignvariableop_21_multi_scale_multi_head_attention_multi_head_attention_11_key_bias:o
Yassignvariableop_22_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel:i
Wassignvariableop_23_multi_scale_multi_head_attention_multi_head_attention_11_value_bias:z
dassignvariableop_24_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel:p
bassignvariableop_25_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias:'
assignvariableop_26_iteration:	 +
!assignvariableop_27_learning_rate: u
_assignvariableop_28_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel:u
_assignvariableop_29_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_kernel:o
]assignvariableop_30_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_bias:o
]assignvariableop_31_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_bias:s
]assignvariableop_32_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel:s
]assignvariableop_33_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_kernel:m
[assignvariableop_34_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_bias:m
[assignvariableop_35_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_bias:u
_assignvariableop_36_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel:u
_assignvariableop_37_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_kernel:o
]assignvariableop_38_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_bias:o
]assignvariableop_39_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_bias:�
jassignvariableop_40_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel:�
jassignvariableop_41_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernel:v
hassignvariableop_42_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias:v
hassignvariableop_43_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_bias:v
`assignvariableop_44_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel:v
`assignvariableop_45_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_kernel:p
^assignvariableop_46_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_bias:p
^assignvariableop_47_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_bias:t
^assignvariableop_48_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel:t
^assignvariableop_49_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_kernel:n
\assignvariableop_50_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_bias:n
\assignvariableop_51_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_bias:v
`assignvariableop_52_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel:v
`assignvariableop_53_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_kernel:p
^assignvariableop_54_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_bias:p
^assignvariableop_55_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_bias:�
kassignvariableop_56_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel:�
kassignvariableop_57_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernel:w
iassignvariableop_58_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias:w
iassignvariableop_59_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_bias:v
`assignvariableop_60_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel:v
`assignvariableop_61_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_kernel:p
^assignvariableop_62_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_bias:p
^assignvariableop_63_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_bias:t
^assignvariableop_64_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel:t
^assignvariableop_65_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_kernel:n
\assignvariableop_66_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_bias:n
\assignvariableop_67_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_bias:v
`assignvariableop_68_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel:v
`assignvariableop_69_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_kernel:p
^assignvariableop_70_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_bias:p
^assignvariableop_71_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_bias:�
kassignvariableop_72_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel:�
kassignvariableop_73_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernel:w
iassignvariableop_74_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias:w
iassignvariableop_75_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_bias:<
)assignvariableop_76_adam_m_dense_3_kernel:	�<
)assignvariableop_77_adam_v_dense_3_kernel:	�5
'assignvariableop_78_adam_m_dense_3_bias:5
'assignvariableop_79_adam_v_dense_3_bias:#
assignvariableop_80_total: #
assignvariableop_81_count: 
identity_83��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_9� 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�
value�B�SB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�
value�B�SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpWassignvariableop_2_multi_scale_multi_head_attention_multi_head_attention_9_query_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpUassignvariableop_3_multi_scale_multi_head_attention_multi_head_attention_9_query_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpUassignvariableop_4_multi_scale_multi_head_attention_multi_head_attention_9_key_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpSassignvariableop_5_multi_scale_multi_head_attention_multi_head_attention_9_key_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpWassignvariableop_6_multi_scale_multi_head_attention_multi_head_attention_9_value_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpUassignvariableop_7_multi_scale_multi_head_attention_multi_head_attention_9_value_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpbassignvariableop_8_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp`assignvariableop_9_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpYassignvariableop_10_multi_scale_multi_head_attention_multi_head_attention_10_query_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpWassignvariableop_11_multi_scale_multi_head_attention_multi_head_attention_10_query_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpWassignvariableop_12_multi_scale_multi_head_attention_multi_head_attention_10_key_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpUassignvariableop_13_multi_scale_multi_head_attention_multi_head_attention_10_key_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpYassignvariableop_14_multi_scale_multi_head_attention_multi_head_attention_10_value_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpWassignvariableop_15_multi_scale_multi_head_attention_multi_head_attention_10_value_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpdassignvariableop_16_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpbassignvariableop_17_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpYassignvariableop_18_multi_scale_multi_head_attention_multi_head_attention_11_query_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpWassignvariableop_19_multi_scale_multi_head_attention_multi_head_attention_11_query_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpWassignvariableop_20_multi_scale_multi_head_attention_multi_head_attention_11_key_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpUassignvariableop_21_multi_scale_multi_head_attention_multi_head_attention_11_key_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpYassignvariableop_22_multi_scale_multi_head_attention_multi_head_attention_11_value_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpWassignvariableop_23_multi_scale_multi_head_attention_multi_head_attention_11_value_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpdassignvariableop_24_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpbassignvariableop_25_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_iterationIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp!assignvariableop_27_learning_rateIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp_assignvariableop_28_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp_assignvariableop_29_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp]assignvariableop_30_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_query_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp]assignvariableop_31_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_query_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp]assignvariableop_32_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp]assignvariableop_33_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp[assignvariableop_34_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_key_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp[assignvariableop_35_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_key_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp_assignvariableop_36_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp_assignvariableop_37_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp]assignvariableop_38_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_value_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp]assignvariableop_39_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_value_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpjassignvariableop_40_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpjassignvariableop_41_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOphassignvariableop_42_adam_m_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOphassignvariableop_43_adam_v_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp`assignvariableop_44_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp`assignvariableop_45_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp^assignvariableop_46_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_query_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp^assignvariableop_47_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_query_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp^assignvariableop_48_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp^assignvariableop_49_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp\assignvariableop_50_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_key_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp\assignvariableop_51_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_key_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp`assignvariableop_52_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp`assignvariableop_53_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp^assignvariableop_54_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_value_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp^assignvariableop_55_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_value_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpkassignvariableop_56_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpkassignvariableop_57_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpiassignvariableop_58_adam_m_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpiassignvariableop_59_adam_v_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp`assignvariableop_60_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp`assignvariableop_61_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp^assignvariableop_62_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_query_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp^assignvariableop_63_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_query_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp^assignvariableop_64_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp^assignvariableop_65_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp\assignvariableop_66_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_key_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp\assignvariableop_67_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_key_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp`assignvariableop_68_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp`assignvariableop_69_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp^assignvariableop_70_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_value_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp^assignvariableop_71_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_value_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOpkassignvariableop_72_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOpkassignvariableop_73_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpiassignvariableop_74_adam_m_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpiassignvariableop_75_adam_v_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_m_dense_3_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp)assignvariableop_77_adam_v_dense_3_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp'assignvariableop_78_adam_m_dense_3_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_v_dense_3_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpassignvariableop_80_totalIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOpassignvariableop_81_countIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_83IdentityIdentity_82:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_83Identity_83:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
B__inference_dense_3_layer_call_and_return_conditional_losses_60214

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_multi_scale_multi_head_attention_layer_call_fn_61282

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60146s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
۩
�
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61431

inputsX
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_query_add_readvariableop_resource:V
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:H
6multi_head_attention_9_key_add_readvariableop_resource:X
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_value_add_readvariableop_resource:c
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource:Q
Cmulti_head_attention_9_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_10_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_query_add_readvariableop_resource:W
Amulti_head_attention_10_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_10_key_add_readvariableop_resource:Y
Cmulti_head_attention_10_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_value_add_readvariableop_resource:d
Nmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_10_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_query_add_readvariableop_resource:W
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_11_key_add_readvariableop_resource:Y
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_value_add_readvariableop_resource:d
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_11_attention_output_add_readvariableop_resource:
identity��;multi_head_attention_10/attention_output/add/ReadVariableOp�Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_10/key/add/ReadVariableOp�8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/query/add/ReadVariableOp�:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/value/add/ReadVariableOp�:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp�;multi_head_attention_11/attention_output/add/ReadVariableOp�Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_11/key/add/ReadVariableOp�8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/query/add/ReadVariableOp�:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/value/add/ReadVariableOp�:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp�:multi_head_attention_9/attention_output/add/ReadVariableOp�Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_9/key/add/ReadVariableOp�7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/query/add/ReadVariableOp�9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/value/add/ReadVariableOp�9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp�
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������a
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:����������
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/query/einsum/EinsumEinsuminputsBmulti_head_attention_10/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/query/add/ReadVariableOpReadVariableOp9multi_head_attention_10_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/query/addAddV24multi_head_attention_10/query/einsum/Einsum:output:08multi_head_attention_10/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_10_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_10/key/einsum/EinsumEinsuminputs@multi_head_attention_10/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_10/key/add/ReadVariableOpReadVariableOp7multi_head_attention_10_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_10/key/addAddV22multi_head_attention_10/key/einsum/Einsum:output:06multi_head_attention_10/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/value/einsum/EinsumEinsuminputsBmulti_head_attention_10/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/value/add/ReadVariableOpReadVariableOp9multi_head_attention_10_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/value/addAddV24multi_head_attention_10/value/einsum/Einsum:output:08multi_head_attention_10/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_10/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_10/MulMul%multi_head_attention_10/query/add:z:0&multi_head_attention_10/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_10/einsum/EinsumEinsum#multi_head_attention_10/key/add:z:0multi_head_attention_10/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_10/softmax_1/SoftmaxSoftmax.multi_head_attention_10/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_10/dropout_1/IdentityIdentity3multi_head_attention_10/softmax_1/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_10/einsum_1/EinsumEinsum3multi_head_attention_10/dropout_1/Identity:output:0%multi_head_attention_10/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_10/attention_output/einsum/EinsumEinsum0multi_head_attention_10/einsum_1/Einsum:output:0Mmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_10/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_10_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_10/attention_output/addAddV2?multi_head_attention_10/attention_output/einsum/Einsum:output:0Cmulti_head_attention_10/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_11/softmax_2/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_11/dropout_2/IdentityIdentity3multi_head_attention_11/softmax_2/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_11/einsum_1/EinsumEinsum3multi_head_attention_11/dropout_2/Identity:output:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2/multi_head_attention_9/attention_output/add:z:00multi_head_attention_10/attention_output/add:z:00multi_head_attention_11/attention_output/add:z:0concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	b
IdentityIdentityconcat:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp<^multi_head_attention_10/attention_output/add/ReadVariableOpF^multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_10/key/add/ReadVariableOp9^multi_head_attention_10/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/query/add/ReadVariableOp;^multi_head_attention_10/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/value/add/ReadVariableOp;^multi_head_attention_10/value/einsum/Einsum/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : 2z
;multi_head_attention_10/attention_output/add/ReadVariableOp;multi_head_attention_10/attention_output/add/ReadVariableOp2�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_10/key/add/ReadVariableOp.multi_head_attention_10/key/add/ReadVariableOp2t
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/query/add/ReadVariableOp0multi_head_attention_10/query/add/ReadVariableOp2x
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/value/add/ReadVariableOp0multi_head_attention_10/value/add/ReadVariableOp2x
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�.
 __inference__wrapped_model_60046
input_4�
psequential_3_multi_scale_multi_head_attention_multi_head_attention_9_query_einsum_einsum_readvariableop_resource:x
fsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_query_add_readvariableop_resource:�
nsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_key_einsum_einsum_readvariableop_resource:v
dsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_key_add_readvariableop_resource:�
psequential_3_multi_scale_multi_head_attention_multi_head_attention_9_value_einsum_einsum_readvariableop_resource:x
fsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_value_add_readvariableop_resource:�
{sequential_3_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource:
qsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_add_readvariableop_resource:�
qsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_query_einsum_einsum_readvariableop_resource:y
gsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_query_add_readvariableop_resource:�
osequential_3_multi_scale_multi_head_attention_multi_head_attention_10_key_einsum_einsum_readvariableop_resource:w
esequential_3_multi_scale_multi_head_attention_multi_head_attention_10_key_add_readvariableop_resource:�
qsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_value_einsum_einsum_readvariableop_resource:y
gsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_value_add_readvariableop_resource:�
|sequential_3_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_einsum_einsum_readvariableop_resource:�
rsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_add_readvariableop_resource:�
qsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_query_einsum_einsum_readvariableop_resource:y
gsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_query_add_readvariableop_resource:�
osequential_3_multi_scale_multi_head_attention_multi_head_attention_11_key_einsum_einsum_readvariableop_resource:w
esequential_3_multi_scale_multi_head_attention_multi_head_attention_11_key_add_readvariableop_resource:�
qsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_value_einsum_einsum_readvariableop_resource:y
gsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_value_add_readvariableop_resource:�
|sequential_3_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource:�
rsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_add_readvariableop_resource:F
3sequential_3_dense_3_matmul_readvariableop_resource:	�B
4sequential_3_dense_3_biasadd_readvariableop_resource:
identity��+sequential_3/dense_3/BiasAdd/ReadVariableOp�*sequential_3/dense_3/MatMul/ReadVariableOp�isequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp�ssequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp�\sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp�fsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp�^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp�hsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp�^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp�hsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp�isequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp�ssequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp�\sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp�fsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp�^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp�hsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp�^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp�hsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp�hsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp�rsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp�[sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp�esequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp�]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp�gsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp�]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp�gsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp�
gsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOppsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Xsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/EinsumEinsuminput_4osequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpfsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Nsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/addAddV2asequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum:output:0esequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
esequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOpnsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Vsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/EinsumEinsuminput_4msequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
[sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpdsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Lsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/addAddV2_sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum:output:0csequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
gsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOppsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Xsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/EinsumEinsuminput_4osequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpfsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Nsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/addAddV2asequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum:output:0esequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Jsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
Hsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/MulMulRsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/add:z:0Ssequential_3/multi_scale_multi_head_attention/multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:����������
Rsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/einsum/EinsumEinsumPsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/add:z:0Lsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Tsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/softmax/SoftmaxSoftmax[sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Usequential_3/multi_scale_multi_head_attention/multi_head_attention_9/dropout/IdentityIdentity^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Tsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/einsum_1/EinsumEinsum^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/dropout/Identity:output:0Rsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
rsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp{sequential_3_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
csequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/EinsumEinsum]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/einsum_1/Einsum:output:0zsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpqsequential_3_multi_scale_multi_head_attention_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Ysequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/addAddV2lsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum:output:0psequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOpReadVariableOpqsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Ysequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/EinsumEinsuminput_4psequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOpReadVariableOpgsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Osequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/addAddV2bsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum:output:0fsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
fsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpReadVariableOposequential_3_multi_scale_multi_head_attention_multi_head_attention_10_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Wsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/EinsumEinsuminput_4nsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
\sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpReadVariableOpesequential_3_multi_scale_multi_head_attention_multi_head_attention_10_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Msequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/addAddV2`sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum:output:0dsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOpReadVariableOpqsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Ysequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/EinsumEinsuminput_4psequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOpReadVariableOpgsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Osequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/addAddV2bsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum:output:0fsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Ksequential_3/multi_scale_multi_head_attention/multi_head_attention_10/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
Isequential_3/multi_scale_multi_head_attention/multi_head_attention_10/MulMulSsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/add:z:0Tsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/Mul/y:output:0*
T0*/
_output_shapes
:����������
Ssequential_3/multi_scale_multi_head_attention/multi_head_attention_10/einsum/EinsumEinsumQsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/add:z:0Msequential_3/multi_scale_multi_head_attention/multi_head_attention_10/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Wsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/softmax_1/SoftmaxSoftmax\sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Xsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/dropout_1/IdentityIdentityasequential_3/multi_scale_multi_head_attention/multi_head_attention_10/softmax_1/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Usequential_3/multi_scale_multi_head_attention/multi_head_attention_10/einsum_1/EinsumEinsumasequential_3/multi_scale_multi_head_attention/multi_head_attention_10/dropout_1/Identity:output:0Ssequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
ssequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp|sequential_3_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
dsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/EinsumEinsum^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/einsum_1/Einsum:output:0{sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
isequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOpReadVariableOprsequential_3_multi_scale_multi_head_attention_multi_head_attention_10_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Zsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/addAddV2msequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum:output:0qsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpqsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Ysequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/EinsumEinsuminput_4psequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOpReadVariableOpgsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Osequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/addAddV2bsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum:output:0fsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
fsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOposequential_3_multi_scale_multi_head_attention_multi_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Wsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/EinsumEinsuminput_4nsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
\sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpReadVariableOpesequential_3_multi_scale_multi_head_attention_multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Msequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/addAddV2`sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum:output:0dsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpqsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Ysequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/EinsumEinsuminput_4psequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOpReadVariableOpgsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Osequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/addAddV2bsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum:output:0fsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Ksequential_3/multi_scale_multi_head_attention/multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
Isequential_3/multi_scale_multi_head_attention/multi_head_attention_11/MulMulSsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/add:z:0Tsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:����������
Ssequential_3/multi_scale_multi_head_attention/multi_head_attention_11/einsum/EinsumEinsumQsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/add:z:0Msequential_3/multi_scale_multi_head_attention/multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Wsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/softmax_2/SoftmaxSoftmax\sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Xsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/dropout_2/IdentityIdentityasequential_3/multi_scale_multi_head_attention/multi_head_attention_11/softmax_2/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Usequential_3/multi_scale_multi_head_attention/multi_head_attention_11/einsum_1/EinsumEinsumasequential_3/multi_scale_multi_head_attention/multi_head_attention_11/dropout_2/Identity:output:0Ssequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
ssequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp|sequential_3_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
dsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/EinsumEinsum^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/einsum_1/Einsum:output:0{sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
isequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOprsequential_3_multi_scale_multi_head_attention_multi_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Zsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/addAddV2msequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum:output:0qsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
9sequential_3/multi_scale_multi_head_attention/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
4sequential_3/multi_scale_multi_head_attention/concatConcatV2]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add:z:0^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add:z:0^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add:z:0Bsequential_3/multi_scale_multi_head_attention/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  �
sequential_3/flatten_3/ReshapeReshape=sequential_3/multi_scale_multi_head_attention/concat:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_3/dense_3/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%sequential_3/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOpj^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOpt^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp]^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpg^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp_^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOpi^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp_^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOpi^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOpj^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOpt^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp]^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpg^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp_^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOpi^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp_^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOpi^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpi^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOps^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp\^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpf^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp^^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOph^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp^^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOph^sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2�
isequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOpisequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp2�
ssequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpssequential_3/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp2�
\sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp\sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp2�
fsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpfsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp2�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp2�
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOphsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp2�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp^sequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp2�
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOphsequential_3/multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp2�
isequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOpisequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp2�
ssequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpssequential_3/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2�
\sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp\sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp2�
fsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpfsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp2�
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOphsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2�
^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp^sequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp2�
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOphsequential_3/multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2�
hsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOphsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp2�
rsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOprsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2�
[sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp[sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp2�
esequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpesequential_3/multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2�
]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp2�
gsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpgsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2�
]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp]sequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp2�
gsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpgsequential_3/multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_4
�
�
'__inference_dense_3_layer_call_fn_61547

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_60214o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
۩
�
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60319

inputsX
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_query_add_readvariableop_resource:V
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:H
6multi_head_attention_9_key_add_readvariableop_resource:X
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_value_add_readvariableop_resource:c
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource:Q
Cmulti_head_attention_9_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_10_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_query_add_readvariableop_resource:W
Amulti_head_attention_10_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_10_key_add_readvariableop_resource:Y
Cmulti_head_attention_10_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_value_add_readvariableop_resource:d
Nmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_10_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_query_add_readvariableop_resource:W
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_11_key_add_readvariableop_resource:Y
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_value_add_readvariableop_resource:d
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_11_attention_output_add_readvariableop_resource:
identity��;multi_head_attention_10/attention_output/add/ReadVariableOp�Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_10/key/add/ReadVariableOp�8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/query/add/ReadVariableOp�:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/value/add/ReadVariableOp�:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp�;multi_head_attention_11/attention_output/add/ReadVariableOp�Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_11/key/add/ReadVariableOp�8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/query/add/ReadVariableOp�:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/value/add/ReadVariableOp�:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp�:multi_head_attention_9/attention_output/add/ReadVariableOp�Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_9/key/add/ReadVariableOp�7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/query/add/ReadVariableOp�9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/value/add/ReadVariableOp�9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp�
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������a
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:����������
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/query/einsum/EinsumEinsuminputsBmulti_head_attention_10/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/query/add/ReadVariableOpReadVariableOp9multi_head_attention_10_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/query/addAddV24multi_head_attention_10/query/einsum/Einsum:output:08multi_head_attention_10/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_10_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_10/key/einsum/EinsumEinsuminputs@multi_head_attention_10/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_10/key/add/ReadVariableOpReadVariableOp7multi_head_attention_10_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_10/key/addAddV22multi_head_attention_10/key/einsum/Einsum:output:06multi_head_attention_10/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/value/einsum/EinsumEinsuminputsBmulti_head_attention_10/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/value/add/ReadVariableOpReadVariableOp9multi_head_attention_10_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/value/addAddV24multi_head_attention_10/value/einsum/Einsum:output:08multi_head_attention_10/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_10/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_10/MulMul%multi_head_attention_10/query/add:z:0&multi_head_attention_10/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_10/einsum/EinsumEinsum#multi_head_attention_10/key/add:z:0multi_head_attention_10/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_10/softmax_1/SoftmaxSoftmax.multi_head_attention_10/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_10/dropout_1/IdentityIdentity3multi_head_attention_10/softmax_1/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_10/einsum_1/EinsumEinsum3multi_head_attention_10/dropout_1/Identity:output:0%multi_head_attention_10/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_10/attention_output/einsum/EinsumEinsum0multi_head_attention_10/einsum_1/Einsum:output:0Mmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_10/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_10_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_10/attention_output/addAddV2?multi_head_attention_10/attention_output/einsum/Einsum:output:0Cmulti_head_attention_10/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_11/softmax_2/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_11/dropout_2/IdentityIdentity3multi_head_attention_11/softmax_2/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_11/einsum_1/EinsumEinsum3multi_head_attention_11/dropout_2/Identity:output:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2/multi_head_attention_9/attention_output/add:z:00multi_head_attention_10/attention_output/add:z:00multi_head_attention_11/attention_output/add:z:0concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	b
IdentityIdentityconcat:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp<^multi_head_attention_10/attention_output/add/ReadVariableOpF^multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_10/key/add/ReadVariableOp9^multi_head_attention_10/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/query/add/ReadVariableOp;^multi_head_attention_10/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/value/add/ReadVariableOp;^multi_head_attention_10/value/einsum/Einsum/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : 2z
;multi_head_attention_10/attention_output/add/ReadVariableOp;multi_head_attention_10/attention_output/add/ReadVariableOp2�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_10/key/add/ReadVariableOp.multi_head_attention_10/key/add/ReadVariableOp2t
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/query/add/ReadVariableOp0multi_head_attention_10/query/add/ReadVariableOp2x
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/value/add/ReadVariableOp0multi_head_attention_10/value/add/ReadVariableOp2x
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_60907
input_4
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_60046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_4
�
�
@__inference_multi_scale_multi_head_attention_layer_call_fn_61335

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60319s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_sequential_3_layer_call_fn_60493
input_4
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_60438o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_4
۩
�
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61527

inputsX
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_query_add_readvariableop_resource:V
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:H
6multi_head_attention_9_key_add_readvariableop_resource:X
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_value_add_readvariableop_resource:c
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource:Q
Cmulti_head_attention_9_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_10_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_query_add_readvariableop_resource:W
Amulti_head_attention_10_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_10_key_add_readvariableop_resource:Y
Cmulti_head_attention_10_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_value_add_readvariableop_resource:d
Nmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_10_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_query_add_readvariableop_resource:W
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_11_key_add_readvariableop_resource:Y
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_value_add_readvariableop_resource:d
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_11_attention_output_add_readvariableop_resource:
identity��;multi_head_attention_10/attention_output/add/ReadVariableOp�Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_10/key/add/ReadVariableOp�8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/query/add/ReadVariableOp�:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/value/add/ReadVariableOp�:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp�;multi_head_attention_11/attention_output/add/ReadVariableOp�Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_11/key/add/ReadVariableOp�8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/query/add/ReadVariableOp�:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/value/add/ReadVariableOp�:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp�:multi_head_attention_9/attention_output/add/ReadVariableOp�Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_9/key/add/ReadVariableOp�7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/query/add/ReadVariableOp�9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/value/add/ReadVariableOp�9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp�
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������a
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:����������
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/query/einsum/EinsumEinsuminputsBmulti_head_attention_10/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/query/add/ReadVariableOpReadVariableOp9multi_head_attention_10_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/query/addAddV24multi_head_attention_10/query/einsum/Einsum:output:08multi_head_attention_10/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_10_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_10/key/einsum/EinsumEinsuminputs@multi_head_attention_10/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_10/key/add/ReadVariableOpReadVariableOp7multi_head_attention_10_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_10/key/addAddV22multi_head_attention_10/key/einsum/Einsum:output:06multi_head_attention_10/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/value/einsum/EinsumEinsuminputsBmulti_head_attention_10/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/value/add/ReadVariableOpReadVariableOp9multi_head_attention_10_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/value/addAddV24multi_head_attention_10/value/einsum/Einsum:output:08multi_head_attention_10/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_10/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_10/MulMul%multi_head_attention_10/query/add:z:0&multi_head_attention_10/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_10/einsum/EinsumEinsum#multi_head_attention_10/key/add:z:0multi_head_attention_10/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_10/softmax_1/SoftmaxSoftmax.multi_head_attention_10/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_10/dropout_1/IdentityIdentity3multi_head_attention_10/softmax_1/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_10/einsum_1/EinsumEinsum3multi_head_attention_10/dropout_1/Identity:output:0%multi_head_attention_10/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_10/attention_output/einsum/EinsumEinsum0multi_head_attention_10/einsum_1/Einsum:output:0Mmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_10/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_10_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_10/attention_output/addAddV2?multi_head_attention_10/attention_output/einsum/Einsum:output:0Cmulti_head_attention_10/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_11/softmax_2/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_11/dropout_2/IdentityIdentity3multi_head_attention_11/softmax_2/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_11/einsum_1/EinsumEinsum3multi_head_attention_11/dropout_2/Identity:output:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2/multi_head_attention_9/attention_output/add:z:00multi_head_attention_10/attention_output/add:z:00multi_head_attention_11/attention_output/add:z:0concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	b
IdentityIdentityconcat:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp<^multi_head_attention_10/attention_output/add/ReadVariableOpF^multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_10/key/add/ReadVariableOp9^multi_head_attention_10/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/query/add/ReadVariableOp;^multi_head_attention_10/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/value/add/ReadVariableOp;^multi_head_attention_10/value/einsum/Einsum/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : 2z
;multi_head_attention_10/attention_output/add/ReadVariableOp;multi_head_attention_10/attention_output/add/ReadVariableOp2�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_10/key/add/ReadVariableOp.multi_head_attention_10/key/add/ReadVariableOp2t
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/query/add/ReadVariableOp0multi_head_attention_10/query/add/ReadVariableOp2x
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/value/add/ReadVariableOp0multi_head_attention_10/value/add/ReadVariableOp2x
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_dense_3_layer_call_and_return_conditional_losses_61557

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
۩
�
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60146

inputsX
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_query_add_readvariableop_resource:V
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:H
6multi_head_attention_9_key_add_readvariableop_resource:X
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource:J
8multi_head_attention_9_value_add_readvariableop_resource:c
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource:Q
Cmulti_head_attention_9_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_10_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_query_add_readvariableop_resource:W
Amulti_head_attention_10_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_10_key_add_readvariableop_resource:Y
Cmulti_head_attention_10_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_10_value_add_readvariableop_resource:d
Nmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_10_attention_output_add_readvariableop_resource:Y
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_query_add_readvariableop_resource:W
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource:I
7multi_head_attention_11_key_add_readvariableop_resource:Y
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource:K
9multi_head_attention_11_value_add_readvariableop_resource:d
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource:R
Dmulti_head_attention_11_attention_output_add_readvariableop_resource:
identity��;multi_head_attention_10/attention_output/add/ReadVariableOp�Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_10/key/add/ReadVariableOp�8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/query/add/ReadVariableOp�:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_10/value/add/ReadVariableOp�:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp�;multi_head_attention_11/attention_output/add/ReadVariableOp�Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp�.multi_head_attention_11/key/add/ReadVariableOp�8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/query/add/ReadVariableOp�:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp�0multi_head_attention_11/value/add/ReadVariableOp�:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp�:multi_head_attention_9/attention_output/add/ReadVariableOp�Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_9/key/add/ReadVariableOp�7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/query/add/ReadVariableOp�9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_9/value/add/ReadVariableOp�9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp�
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������a
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:����������
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/query/einsum/EinsumEinsuminputsBmulti_head_attention_10/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/query/add/ReadVariableOpReadVariableOp9multi_head_attention_10_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/query/addAddV24multi_head_attention_10/query/einsum/Einsum:output:08multi_head_attention_10/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_10_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_10/key/einsum/EinsumEinsuminputs@multi_head_attention_10/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_10/key/add/ReadVariableOpReadVariableOp7multi_head_attention_10_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_10/key/addAddV22multi_head_attention_10/key/einsum/Einsum:output:06multi_head_attention_10/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_10_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_10/value/einsum/EinsumEinsuminputsBmulti_head_attention_10/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_10/value/add/ReadVariableOpReadVariableOp9multi_head_attention_10_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_10/value/addAddV24multi_head_attention_10/value/einsum/Einsum:output:08multi_head_attention_10/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_10/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_10/MulMul%multi_head_attention_10/query/add:z:0&multi_head_attention_10/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_10/einsum/EinsumEinsum#multi_head_attention_10/key/add:z:0multi_head_attention_10/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_10/softmax_1/SoftmaxSoftmax.multi_head_attention_10/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_10/dropout_1/IdentityIdentity3multi_head_attention_10/softmax_1/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_10/einsum_1/EinsumEinsum3multi_head_attention_10/dropout_1/Identity:output:0%multi_head_attention_10/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_10_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_10/attention_output/einsum/EinsumEinsum0multi_head_attention_10/einsum_1/Einsum:output:0Mmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_10/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_10_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_10/attention_output/addAddV2?multi_head_attention_10/attention_output/einsum/Einsum:output:0Cmulti_head_attention_10/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
)multi_head_attention_11/softmax_2/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
*multi_head_attention_11/dropout_2/IdentityIdentity3multi_head_attention_11/softmax_2/Softmax:softmax:0*
T0*/
_output_shapes
:����������
'multi_head_attention_11/einsum_1/EinsumEinsum3multi_head_attention_11/dropout_2/Identity:output:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2/multi_head_attention_9/attention_output/add:z:00multi_head_attention_10/attention_output/add:z:00multi_head_attention_11/attention_output/add:z:0concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	b
IdentityIdentityconcat:output:0^NoOp*
T0*+
_output_shapes
:���������	�
NoOpNoOp<^multi_head_attention_10/attention_output/add/ReadVariableOpF^multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_10/key/add/ReadVariableOp9^multi_head_attention_10/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/query/add/ReadVariableOp;^multi_head_attention_10/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_10/value/add/ReadVariableOp;^multi_head_attention_10/value/einsum/Einsum/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : 2z
;multi_head_attention_10/attention_output/add/ReadVariableOp;multi_head_attention_10/attention_output/add/ReadVariableOp2�
Emulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_10/key/add/ReadVariableOp.multi_head_attention_10/key/add/ReadVariableOp2t
8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp8multi_head_attention_10/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/query/add/ReadVariableOp0multi_head_attention_10/query/add/ReadVariableOp2x
:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp:multi_head_attention_10/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_10/value/add/ReadVariableOp0multi_head_attention_10/value/add/ReadVariableOp2x
:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp:multi_head_attention_10/value/einsum/Einsum/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2�
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60554

inputs<
&multi_scale_multi_head_attention_60498:8
&multi_scale_multi_head_attention_60500:<
&multi_scale_multi_head_attention_60502:8
&multi_scale_multi_head_attention_60504:<
&multi_scale_multi_head_attention_60506:8
&multi_scale_multi_head_attention_60508:<
&multi_scale_multi_head_attention_60510:4
&multi_scale_multi_head_attention_60512:<
&multi_scale_multi_head_attention_60514:8
&multi_scale_multi_head_attention_60516:<
&multi_scale_multi_head_attention_60518:8
&multi_scale_multi_head_attention_60520:<
&multi_scale_multi_head_attention_60522:8
&multi_scale_multi_head_attention_60524:<
&multi_scale_multi_head_attention_60526:4
&multi_scale_multi_head_attention_60528:<
&multi_scale_multi_head_attention_60530:8
&multi_scale_multi_head_attention_60532:<
&multi_scale_multi_head_attention_60534:8
&multi_scale_multi_head_attention_60536:<
&multi_scale_multi_head_attention_60538:8
&multi_scale_multi_head_attention_60540:<
&multi_scale_multi_head_attention_60542:4
&multi_scale_multi_head_attention_60544: 
dense_3_60548:	�
dense_3_60550:
identity��dense_3/StatefulPartitionedCall�8multi_scale_multi_head_attention/StatefulPartitionedCall�

8multi_scale_multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinputs&multi_scale_multi_head_attention_60498&multi_scale_multi_head_attention_60500&multi_scale_multi_head_attention_60502&multi_scale_multi_head_attention_60504&multi_scale_multi_head_attention_60506&multi_scale_multi_head_attention_60508&multi_scale_multi_head_attention_60510&multi_scale_multi_head_attention_60512&multi_scale_multi_head_attention_60514&multi_scale_multi_head_attention_60516&multi_scale_multi_head_attention_60518&multi_scale_multi_head_attention_60520&multi_scale_multi_head_attention_60522&multi_scale_multi_head_attention_60524&multi_scale_multi_head_attention_60526&multi_scale_multi_head_attention_60528&multi_scale_multi_head_attention_60530&multi_scale_multi_head_attention_60532&multi_scale_multi_head_attention_60534&multi_scale_multi_head_attention_60536&multi_scale_multi_head_attention_60538&multi_scale_multi_head_attention_60540&multi_scale_multi_head_attention_60542&multi_scale_multi_head_attention_60544*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_60319�
flatten_3/PartitionedCallPartitionedCallAmulti_scale_multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_60202�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_60548dense_3_60550*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_60214w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_3/StatefulPartitionedCall9^multi_scale_multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2t
8multi_scale_multi_head_attention/StatefulPartitionedCall8multi_scale_multi_head_attention/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
E
)__inference_flatten_3_layer_call_fn_61532

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_60202a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_60202

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�)
G__inference_sequential_3_layer_call_and_return_conditional_losses_61125

inputsy
cmulti_scale_multi_head_attention_multi_head_attention_9_query_einsum_einsum_readvariableop_resource:k
Ymulti_scale_multi_head_attention_multi_head_attention_9_query_add_readvariableop_resource:w
amulti_scale_multi_head_attention_multi_head_attention_9_key_einsum_einsum_readvariableop_resource:i
Wmulti_scale_multi_head_attention_multi_head_attention_9_key_add_readvariableop_resource:y
cmulti_scale_multi_head_attention_multi_head_attention_9_value_einsum_einsum_readvariableop_resource:k
Ymulti_scale_multi_head_attention_multi_head_attention_9_value_add_readvariableop_resource:�
nmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource:r
dmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_10_query_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_10_query_add_readvariableop_resource:x
bmulti_scale_multi_head_attention_multi_head_attention_10_key_einsum_einsum_readvariableop_resource:j
Xmulti_scale_multi_head_attention_multi_head_attention_10_key_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_10_value_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_10_value_add_readvariableop_resource:�
omulti_scale_multi_head_attention_multi_head_attention_10_attention_output_einsum_einsum_readvariableop_resource:s
emulti_scale_multi_head_attention_multi_head_attention_10_attention_output_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_11_query_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_11_query_add_readvariableop_resource:x
bmulti_scale_multi_head_attention_multi_head_attention_11_key_einsum_einsum_readvariableop_resource:j
Xmulti_scale_multi_head_attention_multi_head_attention_11_key_add_readvariableop_resource:z
dmulti_scale_multi_head_attention_multi_head_attention_11_value_einsum_einsum_readvariableop_resource:l
Zmulti_scale_multi_head_attention_multi_head_attention_11_value_add_readvariableop_resource:�
omulti_scale_multi_head_attention_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource:s
emulti_scale_multi_head_attention_multi_head_attention_11_attention_output_add_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	�5
'dense_3_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp�fmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp�Omulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp�Ymulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp�\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp�fmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp�Omulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp�Ymulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp�Qmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp�[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp�emulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp�Nmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp�Xmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp�Pmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp�Zmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp�Pmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp�Zmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp�
Zmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpcmulti_scale_multi_head_attention_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Kmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/EinsumEinsuminputsbmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Pmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpYmulti_scale_multi_head_attention_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Amulti_scale_multi_head_attention/multi_head_attention_9/query/addAddV2Tmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum:output:0Xmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Xmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOpamulti_scale_multi_head_attention_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Imulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/EinsumEinsuminputs`multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Nmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpWmulti_scale_multi_head_attention_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
?multi_scale_multi_head_attention/multi_head_attention_9/key/addAddV2Rmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum:output:0Vmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Zmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpcmulti_scale_multi_head_attention_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Kmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/EinsumEinsuminputsbmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Pmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpYmulti_scale_multi_head_attention_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Amulti_scale_multi_head_attention/multi_head_attention_9/value/addAddV2Tmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum:output:0Xmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
=multi_scale_multi_head_attention/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
;multi_scale_multi_head_attention/multi_head_attention_9/MulMulEmulti_scale_multi_head_attention/multi_head_attention_9/query/add:z:0Fmulti_scale_multi_head_attention/multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:����������
Emulti_scale_multi_head_attention/multi_head_attention_9/einsum/EinsumEinsumCmulti_scale_multi_head_attention/multi_head_attention_9/key/add:z:0?multi_scale_multi_head_attention/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Gmulti_scale_multi_head_attention/multi_head_attention_9/softmax/SoftmaxSoftmaxNmulti_scale_multi_head_attention/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Hmulti_scale_multi_head_attention/multi_head_attention_9/dropout/IdentityIdentityQmulti_scale_multi_head_attention/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Gmulti_scale_multi_head_attention/multi_head_attention_9/einsum_1/EinsumEinsumQmulti_scale_multi_head_attention/multi_head_attention_9/dropout/Identity:output:0Emulti_scale_multi_head_attention/multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
emulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpnmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Vmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/EinsumEinsumPmulti_scale_multi_head_attention/multi_head_attention_9/einsum_1/Einsum:output:0mmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/addAddV2_multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum:output:0cmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_10_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_10/query/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_10_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_10/query/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Ymulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpReadVariableOpbmulti_scale_multi_head_attention_multi_head_attention_10_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Jmulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/EinsumEinsuminputsamulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Omulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpReadVariableOpXmulti_scale_multi_head_attention_multi_head_attention_10_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
@multi_scale_multi_head_attention/multi_head_attention_10/key/addAddV2Smulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum:output:0Wmulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_10_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_10/value/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_10_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_10/value/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
>multi_scale_multi_head_attention/multi_head_attention_10/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
<multi_scale_multi_head_attention/multi_head_attention_10/MulMulFmulti_scale_multi_head_attention/multi_head_attention_10/query/add:z:0Gmulti_scale_multi_head_attention/multi_head_attention_10/Mul/y:output:0*
T0*/
_output_shapes
:����������
Fmulti_scale_multi_head_attention/multi_head_attention_10/einsum/EinsumEinsumDmulti_scale_multi_head_attention/multi_head_attention_10/key/add:z:0@multi_scale_multi_head_attention/multi_head_attention_10/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Jmulti_scale_multi_head_attention/multi_head_attention_10/softmax_1/SoftmaxSoftmaxOmulti_scale_multi_head_attention/multi_head_attention_10/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Kmulti_scale_multi_head_attention/multi_head_attention_10/dropout_1/IdentityIdentityTmulti_scale_multi_head_attention/multi_head_attention_10/softmax_1/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Hmulti_scale_multi_head_attention/multi_head_attention_10/einsum_1/EinsumEinsumTmulti_scale_multi_head_attention/multi_head_attention_10/dropout_1/Identity:output:0Fmulti_scale_multi_head_attention/multi_head_attention_10/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
fmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpomulti_scale_multi_head_attention_multi_head_attention_10_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Wmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/EinsumEinsumQmulti_scale_multi_head_attention/multi_head_attention_10/einsum_1/Einsum:output:0nmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOpReadVariableOpemulti_scale_multi_head_attention_multi_head_attention_10_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Mmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/addAddV2`multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum:output:0dmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_11/query/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_11/query/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Ymulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpbmulti_scale_multi_head_attention_multi_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Jmulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/EinsumEinsuminputsamulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Omulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpReadVariableOpXmulti_scale_multi_head_attention_multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
@multi_scale_multi_head_attention/multi_head_attention_11/key/addAddV2Smulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum:output:0Wmulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpdmulti_scale_multi_head_attention_multi_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Lmulti_scale_multi_head_attention/multi_head_attention_11/value/einsum/EinsumEinsuminputscmulti_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������*
equationabc,cde->abde�
Qmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOpReadVariableOpZmulti_scale_multi_head_attention_multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmulti_scale_multi_head_attention/multi_head_attention_11/value/addAddV2Umulti_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum:output:0Ymulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
>multi_scale_multi_head_attention/multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
<multi_scale_multi_head_attention/multi_head_attention_11/MulMulFmulti_scale_multi_head_attention/multi_head_attention_11/query/add:z:0Gmulti_scale_multi_head_attention/multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:����������
Fmulti_scale_multi_head_attention/multi_head_attention_11/einsum/EinsumEinsumDmulti_scale_multi_head_attention/multi_head_attention_11/key/add:z:0@multi_scale_multi_head_attention/multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Jmulti_scale_multi_head_attention/multi_head_attention_11/softmax_2/SoftmaxSoftmaxOmulti_scale_multi_head_attention/multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Kmulti_scale_multi_head_attention/multi_head_attention_11/dropout_2/IdentityIdentityTmulti_scale_multi_head_attention/multi_head_attention_11/softmax_2/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Hmulti_scale_multi_head_attention/multi_head_attention_11/einsum_1/EinsumEinsumTmulti_scale_multi_head_attention/multi_head_attention_11/dropout_2/Identity:output:0Fmulti_scale_multi_head_attention/multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:���������*
equationacbe,aecd->abcd�
fmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpomulti_scale_multi_head_attention_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
Wmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/EinsumEinsumQmulti_scale_multi_head_attention/multi_head_attention_11/einsum_1/Einsum:output:0nmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpemulti_scale_multi_head_attention_multi_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
Mmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/addAddV2`multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum:output:0dmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w
,multi_scale_multi_head_attention/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
'multi_scale_multi_head_attention/concatConcatV2Pmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/add:z:0Qmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/add:z:0Qmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/add:z:05multi_scale_multi_head_attention/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  �
flatten_3/ReshapeReshape0multi_scale_multi_head_attention/concat:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp]^multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOpg^multi_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpP^multi_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpZ^multi_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp]^multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOpg^multi_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpP^multi_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpZ^multi_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpR^multi_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp\^multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOpf^multi_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpO^multi_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpY^multi_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpQ^multi_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp[^multi_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpQ^multi_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp[^multi_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2�
\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp\multi_scale_multi_head_attention/multi_head_attention_10/attention_output/add/ReadVariableOp2�
fmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOpfmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/einsum/Einsum/ReadVariableOp2�
Omulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOpOmulti_scale_multi_head_attention/multi_head_attention_10/key/add/ReadVariableOp2�
Ymulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOpYmulti_scale_multi_head_attention/multi_head_attention_10/key/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_10/query/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_10/query/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_10/value/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_10/value/einsum/Einsum/ReadVariableOp2�
\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp\multi_scale_multi_head_attention/multi_head_attention_11/attention_output/add/ReadVariableOp2�
fmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpfmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2�
Omulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOpOmulti_scale_multi_head_attention/multi_head_attention_11/key/add/ReadVariableOp2�
Ymulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpYmulti_scale_multi_head_attention/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_11/query/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2�
Qmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOpQmulti_scale_multi_head_attention/multi_head_attention_11/value/add/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2�
[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp[multi_scale_multi_head_attention/multi_head_attention_9/attention_output/add/ReadVariableOp2�
emulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpemulti_scale_multi_head_attention/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2�
Nmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOpNmulti_scale_multi_head_attention/multi_head_attention_9/key/add/ReadVariableOp2�
Xmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpXmulti_scale_multi_head_attention/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2�
Pmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOpPmulti_scale_multi_head_attention/multi_head_attention_9/query/add/ReadVariableOp2�
Zmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpZmulti_scale_multi_head_attention/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2�
Pmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOpPmulti_scale_multi_head_attention/multi_head_attention_9/value/add/ReadVariableOp2�
Zmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpZmulti_scale_multi_head_attention/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_61538

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_44
serving_default_input_4:0���������;
dense_30
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
attention_layers"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
 24
!25"
trackable_list_wrapper
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
 24
!25"
trackable_list_wrapper
 "
trackable_list_wrapper
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
?trace_0
@trace_1
Atrace_2
Btrace_32�
,__inference_sequential_3_layer_call_fn_60493
,__inference_sequential_3_layer_call_fn_60609
,__inference_sequential_3_layer_call_fn_60964
,__inference_sequential_3_layer_call_fn_61021�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z?trace_0z@trace_1zAtrace_2zBtrace_3
�
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60221
G__inference_sequential_3_layer_call_and_return_conditional_losses_60376
G__inference_sequential_3_layer_call_and_return_conditional_losses_61125
G__inference_sequential_3_layer_call_and_return_conditional_losses_61229�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
�B�
 __inference__wrapped_model_60046input_4"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
G
_variables
H_iterations
I_learning_rate
J_index_dict
K
_momentums
L_velocities
M_update_step_xla"
experimentalOptimizer
,
Nserving_default"
signature_map
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923"
trackable_list_wrapper
�
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ttrace_0
Utrace_12�
@__inference_multi_scale_multi_head_attention_layer_call_fn_61282
@__inference_multi_scale_multi_head_attention_layer_call_fn_61335�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 zTtrace_0zUtrace_1
�
Vtrace_0
Wtrace_12�
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61431
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61527�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 zVtrace_0zWtrace_1
5
X0
Y1
Z2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_02�
)__inference_flatten_3_layer_call_fn_61532�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
�
atrace_02�
D__inference_flatten_3_layer_call_and_return_conditional_losses_61538�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
gtrace_02�
'__inference_dense_3_layer_call_fn_61547�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0
�
htrace_02�
B__inference_dense_3_layer_call_and_return_conditional_losses_61557�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
!:	�2dense_3/kernel
:2dense_3/bias
Z:X2Dmulti_scale_multi_head_attention/multi_head_attention_9/query/kernel
T:R2Bmulti_scale_multi_head_attention/multi_head_attention_9/query/bias
X:V2Bmulti_scale_multi_head_attention/multi_head_attention_9/key/kernel
R:P2@multi_scale_multi_head_attention/multi_head_attention_9/key/bias
Z:X2Dmulti_scale_multi_head_attention/multi_head_attention_9/value/kernel
T:R2Bmulti_scale_multi_head_attention/multi_head_attention_9/value/bias
e:c2Omulti_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel
[:Y2Mmulti_scale_multi_head_attention/multi_head_attention_9/attention_output/bias
[:Y2Emulti_scale_multi_head_attention/multi_head_attention_10/query/kernel
U:S2Cmulti_scale_multi_head_attention/multi_head_attention_10/query/bias
Y:W2Cmulti_scale_multi_head_attention/multi_head_attention_10/key/kernel
S:Q2Amulti_scale_multi_head_attention/multi_head_attention_10/key/bias
[:Y2Emulti_scale_multi_head_attention/multi_head_attention_10/value/kernel
U:S2Cmulti_scale_multi_head_attention/multi_head_attention_10/value/bias
f:d2Pmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel
\:Z2Nmulti_scale_multi_head_attention/multi_head_attention_10/attention_output/bias
[:Y2Emulti_scale_multi_head_attention/multi_head_attention_11/query/kernel
U:S2Cmulti_scale_multi_head_attention/multi_head_attention_11/query/bias
Y:W2Cmulti_scale_multi_head_attention/multi_head_attention_11/key/kernel
S:Q2Amulti_scale_multi_head_attention/multi_head_attention_11/key/bias
[:Y2Emulti_scale_multi_head_attention/multi_head_attention_11/value/kernel
U:S2Cmulti_scale_multi_head_attention/multi_head_attention_11/value/bias
f:d2Pmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel
\:Z2Nmulti_scale_multi_head_attention/multi_head_attention_11/attention_output/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_3_layer_call_fn_60493input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_3_layer_call_fn_60609input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_3_layer_call_fn_60964inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_3_layer_call_fn_61021inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60221input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60376input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_61125inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_61229inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
H0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15
y16
z17
{18
|19
}20
~21
22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
j0
l1
n2
p3
r4
t5
v6
x7
z8
|9
~10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25"
trackable_list_wrapper
�
k0
m1
o2
q3
s4
u5
w6
y7
{8
}9
10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_60907input_4"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
5
X0
Y1
Z2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
@__inference_multi_scale_multi_head_attention_layer_call_fn_61282inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
@__inference_multi_scale_multi_head_attention_layer_call_fn_61335inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61431inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61527inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
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
�B�
)__inference_flatten_3_layer_call_fn_61532inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_3_layer_call_and_return_conditional_losses_61538inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
'__inference_dense_3_layer_call_fn_61547inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_3_layer_call_and_return_conditional_losses_61557inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
_:]2KAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel
_:]2KAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/kernel
Y:W2IAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/query/bias
Y:W2IAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/query/bias
]:[2IAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel
]:[2IAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/kernel
W:U2GAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/key/bias
W:U2GAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/key/bias
_:]2KAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel
_:]2KAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/kernel
Y:W2IAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/value/bias
Y:W2IAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/value/bias
j:h2VAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel
j:h2VAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/kernel
`:^2TAdam/m/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias
`:^2TAdam/v/multi_scale_multi_head_attention/multi_head_attention_9/attention_output/bias
`:^2LAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel
`:^2LAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/kernel
Z:X2JAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/query/bias
Z:X2JAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/query/bias
^:\2JAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel
^:\2JAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/kernel
X:V2HAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/key/bias
X:V2HAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/key/bias
`:^2LAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel
`:^2LAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/kernel
Z:X2JAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/value/bias
Z:X2JAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/value/bias
k:i2WAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel
k:i2WAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/kernel
a:_2UAdam/m/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias
a:_2UAdam/v/multi_scale_multi_head_attention/multi_head_attention_10/attention_output/bias
`:^2LAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel
`:^2LAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/kernel
Z:X2JAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/query/bias
Z:X2JAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/query/bias
^:\2JAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel
^:\2JAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/kernel
X:V2HAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/key/bias
X:V2HAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/key/bias
`:^2LAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel
`:^2LAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/kernel
Z:X2JAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/value/bias
Z:X2JAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/value/bias
k:i2WAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel
k:i2WAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/kernel
a:_2UAdam/m/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias
a:_2UAdam/v/multi_scale_multi_head_attention/multi_head_attention_11/attention_output/bias
&:$	�2Adam/m/dense_3/kernel
&:$	�2Adam/v/dense_3/kernel
:2Adam/m/dense_3/bias
:2Adam/v/dense_3/bias
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

"kernel
#bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

$kernel
%bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

&kernel
'bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

(kernel
)bias"
_tf_keras_layer
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

*kernel
+bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

,kernel
-bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

.kernel
/bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

0kernel
1bias"
_tf_keras_layer
X
20
31
42
53
64
75
86
97"
trackable_list_wrapper
X
20
31
42
53
64
75
86
97"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

2kernel
3bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

4kernel
5bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

6kernel
7bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

8kernel
9bias"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_dict_wrapper�
 __inference__wrapped_model_60046�"#$%&'()*+,-./0123456789 !4�1
*�'
%�"
input_4���������
� "1�.
,
dense_3!�
dense_3����������
B__inference_dense_3_layer_call_and_return_conditional_losses_61557d !0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
'__inference_dense_3_layer_call_fn_61547Y !0�-
&�#
!�
inputs����������
� "!�
unknown����������
D__inference_flatten_3_layer_call_and_return_conditional_losses_61538d3�0
)�&
$�!
inputs���������	
� "-�*
#� 
tensor_0����������
� �
)__inference_flatten_3_layer_call_fn_61532Y3�0
)�&
$�!
inputs���������	
� ""�
unknown�����������
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61431�"#$%&'()*+,-./0123456789C�@
)�&
$�!
inputs���������
�

trainingp"0�-
&�#
tensor_0���������	
� �
[__inference_multi_scale_multi_head_attention_layer_call_and_return_conditional_losses_61527�"#$%&'()*+,-./0123456789C�@
)�&
$�!
inputs���������
�

trainingp "0�-
&�#
tensor_0���������	
� �
@__inference_multi_scale_multi_head_attention_layer_call_fn_61282�"#$%&'()*+,-./0123456789C�@
)�&
$�!
inputs���������
�

trainingp"%�"
unknown���������	�
@__inference_multi_scale_multi_head_attention_layer_call_fn_61335�"#$%&'()*+,-./0123456789C�@
)�&
$�!
inputs���������
�

trainingp "%�"
unknown���������	�
G__inference_sequential_3_layer_call_and_return_conditional_losses_60221�"#$%&'()*+,-./0123456789 !<�9
2�/
%�"
input_4���������
p

 
� ",�)
"�
tensor_0���������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_60376�"#$%&'()*+,-./0123456789 !<�9
2�/
%�"
input_4���������
p 

 
� ",�)
"�
tensor_0���������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_61125�"#$%&'()*+,-./0123456789 !;�8
1�.
$�!
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_61229�"#$%&'()*+,-./0123456789 !;�8
1�.
$�!
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
,__inference_sequential_3_layer_call_fn_60493}"#$%&'()*+,-./0123456789 !<�9
2�/
%�"
input_4���������
p

 
� "!�
unknown����������
,__inference_sequential_3_layer_call_fn_60609}"#$%&'()*+,-./0123456789 !<�9
2�/
%�"
input_4���������
p 

 
� "!�
unknown����������
,__inference_sequential_3_layer_call_fn_60964|"#$%&'()*+,-./0123456789 !;�8
1�.
$�!
inputs���������
p

 
� "!�
unknown����������
,__inference_sequential_3_layer_call_fn_61021|"#$%&'()*+,-./0123456789 !;�8
1�.
$�!
inputs���������
p 

 
� "!�
unknown����������
#__inference_signature_wrapper_60907�"#$%&'()*+,-./0123456789 !?�<
� 
5�2
0
input_4%�"
input_4���������"1�.
,
dense_3!�
dense_3���������