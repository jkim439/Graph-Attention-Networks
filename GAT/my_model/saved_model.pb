??2
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
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
;
Elu
features"T
activations"T"
Ttype:
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
]
SparseSoftmax

sp_indices	
	sp_values"T
sp_shape	
output"T"
Ttype:
2
?
SparseTensorDenseMatMul
	a_indices"Tindices
a_values"T
a_shape	
b"T
product"T"	
Ttype"
Tindicestype0	:
2	"
	adjoint_abool( "
	adjoint_bbool( 
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??.
?
graph_attention_sparse/ig_deltaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!graph_attention_sparse/ig_delta
?
3graph_attention_sparse/ig_delta/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/ig_delta*
_output_shapes
: *
dtype0
?
(graph_attention_sparse/ig_non_exist_edgeVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(graph_attention_sparse/ig_non_exist_edge
?
<graph_attention_sparse/ig_non_exist_edge/Read/ReadVariableOpReadVariableOp(graph_attention_sparse/ig_non_exist_edge*
_output_shapes
: *
dtype0
?
graph_attention_sparse/kernel_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_0
?
3graph_attention_sparse/kernel_0/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_0*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_0
?
1graph_attention_sparse/bias_0/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_0*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_0VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_0
?
=graph_attention_sparse/attn_kernel_self_0/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_0*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_0VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_0
?
>graph_attention_sparse/attn_kernel_neigh_0/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_0*
_output_shapes

:*
dtype0
?
graph_attention_sparse/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_1
?
3graph_attention_sparse/kernel_1/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_1*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_1
?
1graph_attention_sparse/bias_1/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_1*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_1
?
=graph_attention_sparse/attn_kernel_self_1/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_1*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_1
?
>graph_attention_sparse/attn_kernel_neigh_1/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_1*
_output_shapes

:*
dtype0
?
graph_attention_sparse/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_2
?
3graph_attention_sparse/kernel_2/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_2*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_2
?
1graph_attention_sparse/bias_2/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_2*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_2
?
=graph_attention_sparse/attn_kernel_self_2/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_2*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_2
?
>graph_attention_sparse/attn_kernel_neigh_2/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_2*
_output_shapes

:*
dtype0
?
graph_attention_sparse/kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_3
?
3graph_attention_sparse/kernel_3/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_3*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_3
?
1graph_attention_sparse/bias_3/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_3*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_3
?
=graph_attention_sparse/attn_kernel_self_3/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_3*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_3
?
>graph_attention_sparse/attn_kernel_neigh_3/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_3*
_output_shapes

:*
dtype0
?
graph_attention_sparse/kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_4
?
3graph_attention_sparse/kernel_4/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_4*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_4
?
1graph_attention_sparse/bias_4/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_4*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_4VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_4
?
=graph_attention_sparse/attn_kernel_self_4/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_4*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_4VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_4
?
>graph_attention_sparse/attn_kernel_neigh_4/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_4*
_output_shapes

:*
dtype0
?
graph_attention_sparse/kernel_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_5
?
3graph_attention_sparse/kernel_5/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_5*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_5
?
1graph_attention_sparse/bias_5/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_5*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_5VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_5
?
=graph_attention_sparse/attn_kernel_self_5/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_5*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_5VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_5
?
>graph_attention_sparse/attn_kernel_neigh_5/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_5*
_output_shapes

:*
dtype0
?
graph_attention_sparse/kernel_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_6
?
3graph_attention_sparse/kernel_6/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_6*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_6
?
1graph_attention_sparse/bias_6/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_6*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_6VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_6
?
=graph_attention_sparse/attn_kernel_self_6/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_6*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_6VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_6
?
>graph_attention_sparse/attn_kernel_neigh_6/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_6*
_output_shapes

:*
dtype0
?
graph_attention_sparse/kernel_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!graph_attention_sparse/kernel_7
?
3graph_attention_sparse/kernel_7/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/kernel_7*
_output_shapes
:	?*
dtype0
?
graph_attention_sparse/bias_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_attention_sparse/bias_7
?
1graph_attention_sparse/bias_7/Read/ReadVariableOpReadVariableOpgraph_attention_sparse/bias_7*
_output_shapes
:*
dtype0
?
)graph_attention_sparse/attn_kernel_self_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)graph_attention_sparse/attn_kernel_self_7
?
=graph_attention_sparse/attn_kernel_self_7/Read/ReadVariableOpReadVariableOp)graph_attention_sparse/attn_kernel_self_7*
_output_shapes

:*
dtype0
?
*graph_attention_sparse/attn_kernel_neigh_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*graph_attention_sparse/attn_kernel_neigh_7
?
>graph_attention_sparse/attn_kernel_neigh_7/Read/ReadVariableOpReadVariableOp*graph_attention_sparse/attn_kernel_neigh_7*
_output_shapes

:*
dtype0
?
!graph_attention_sparse_1/ig_deltaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!graph_attention_sparse_1/ig_delta
?
5graph_attention_sparse_1/ig_delta/Read/ReadVariableOpReadVariableOp!graph_attention_sparse_1/ig_delta*
_output_shapes
: *
dtype0
?
*graph_attention_sparse_1/ig_non_exist_edgeVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*graph_attention_sparse_1/ig_non_exist_edge
?
>graph_attention_sparse_1/ig_non_exist_edge/Read/ReadVariableOpReadVariableOp*graph_attention_sparse_1/ig_non_exist_edge*
_output_shapes
: *
dtype0
?
!graph_attention_sparse_1/kernel_0VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!graph_attention_sparse_1/kernel_0
?
5graph_attention_sparse_1/kernel_0/Read/ReadVariableOpReadVariableOp!graph_attention_sparse_1/kernel_0*
_output_shapes

:@*
dtype0
?
graph_attention_sparse_1/bias_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!graph_attention_sparse_1/bias_0
?
3graph_attention_sparse_1/bias_0/Read/ReadVariableOpReadVariableOpgraph_attention_sparse_1/bias_0*
_output_shapes
:*
dtype0
?
+graph_attention_sparse_1/attn_kernel_self_0VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+graph_attention_sparse_1/attn_kernel_self_0
?
?graph_attention_sparse_1/attn_kernel_self_0/Read/ReadVariableOpReadVariableOp+graph_attention_sparse_1/attn_kernel_self_0*
_output_shapes

:*
dtype0
?
,graph_attention_sparse_1/attn_kernel_neigh_0VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,graph_attention_sparse_1/attn_kernel_neigh_0
?
@graph_attention_sparse_1/attn_kernel_neigh_0/Read/ReadVariableOpReadVariableOp,graph_attention_sparse_1/attn_kernel_neigh_0*
_output_shapes

:*
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
?
&Adam/graph_attention_sparse/kernel_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_0/m
?
:Adam/graph_attention_sparse/kernel_0/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_0/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_0/m
?
8Adam/graph_attention_sparse/bias_0/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_0/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_0/m
?
DAdam/graph_attention_sparse/attn_kernel_self_0/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_0/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_0/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_0/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_0/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_1/m
?
:Adam/graph_attention_sparse/kernel_1/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_1/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_1/m
?
8Adam/graph_attention_sparse/bias_1/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_1/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_1/m
?
DAdam/graph_attention_sparse/attn_kernel_self_1/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_1/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_1/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_1/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_1/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_2/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_2/m
?
:Adam/graph_attention_sparse/kernel_2/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_2/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_2/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_2/m
?
8Adam/graph_attention_sparse/bias_2/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_2/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_2/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_2/m
?
DAdam/graph_attention_sparse/attn_kernel_self_2/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_2/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_2/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_2/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_2/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_2/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_3/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_3/m
?
:Adam/graph_attention_sparse/kernel_3/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_3/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_3/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_3/m
?
8Adam/graph_attention_sparse/bias_3/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_3/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_3/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_3/m
?
DAdam/graph_attention_sparse/attn_kernel_self_3/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_3/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_3/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_3/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_3/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_3/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_4/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_4/m
?
:Adam/graph_attention_sparse/kernel_4/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_4/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_4/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_4/m
?
8Adam/graph_attention_sparse/bias_4/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_4/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_4/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_4/m
?
DAdam/graph_attention_sparse/attn_kernel_self_4/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_4/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_4/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_4/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_4/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_4/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_5/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_5/m
?
:Adam/graph_attention_sparse/kernel_5/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_5/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_5/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_5/m
?
8Adam/graph_attention_sparse/bias_5/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_5/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_5/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_5/m
?
DAdam/graph_attention_sparse/attn_kernel_self_5/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_5/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_5/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_5/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_5/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_5/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_6/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_6/m
?
:Adam/graph_attention_sparse/kernel_6/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_6/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_6/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_6/m
?
8Adam/graph_attention_sparse/bias_6/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_6/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_6/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_6/m
?
DAdam/graph_attention_sparse/attn_kernel_self_6/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_6/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_6/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_6/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_6/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_6/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_7/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_7/m
?
:Adam/graph_attention_sparse/kernel_7/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_7/m*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_7/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_7/m
?
8Adam/graph_attention_sparse/bias_7/m/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_7/m*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_7/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_7/m
?
DAdam/graph_attention_sparse/attn_kernel_self_7/m/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_7/m*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_7/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_7/m
?
EAdam/graph_attention_sparse/attn_kernel_neigh_7/m/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_7/m*
_output_shapes

:*
dtype0
?
(Adam/graph_attention_sparse_1/kernel_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/graph_attention_sparse_1/kernel_0/m
?
<Adam/graph_attention_sparse_1/kernel_0/m/Read/ReadVariableOpReadVariableOp(Adam/graph_attention_sparse_1/kernel_0/m*
_output_shapes

:@*
dtype0
?
&Adam/graph_attention_sparse_1/bias_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/graph_attention_sparse_1/bias_0/m
?
:Adam/graph_attention_sparse_1/bias_0/m/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse_1/bias_0/m*
_output_shapes
:*
dtype0
?
2Adam/graph_attention_sparse_1/attn_kernel_self_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/graph_attention_sparse_1/attn_kernel_self_0/m
?
FAdam/graph_attention_sparse_1/attn_kernel_self_0/m/Read/ReadVariableOpReadVariableOp2Adam/graph_attention_sparse_1/attn_kernel_self_0/m*
_output_shapes

:*
dtype0
?
3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53Adam/graph_attention_sparse_1/attn_kernel_neigh_0/m
?
GAdam/graph_attention_sparse_1/attn_kernel_neigh_0/m/Read/ReadVariableOpReadVariableOp3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/m*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_0/v
?
:Adam/graph_attention_sparse/kernel_0/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_0/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_0/v
?
8Adam/graph_attention_sparse/bias_0/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_0/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_0/v
?
DAdam/graph_attention_sparse/attn_kernel_self_0/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_0/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_0/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_0/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_0/v*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_1/v
?
:Adam/graph_attention_sparse/kernel_1/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_1/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_1/v
?
8Adam/graph_attention_sparse/bias_1/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_1/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_1/v
?
DAdam/graph_attention_sparse/attn_kernel_self_1/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_1/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_1/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_1/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_1/v*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_2/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_2/v
?
:Adam/graph_attention_sparse/kernel_2/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_2/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_2/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_2/v
?
8Adam/graph_attention_sparse/bias_2/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_2/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_2/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_2/v
?
DAdam/graph_attention_sparse/attn_kernel_self_2/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_2/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_2/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_2/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_2/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_2/v*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_3/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_3/v
?
:Adam/graph_attention_sparse/kernel_3/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_3/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_3/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_3/v
?
8Adam/graph_attention_sparse/bias_3/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_3/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_3/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_3/v
?
DAdam/graph_attention_sparse/attn_kernel_self_3/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_3/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_3/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_3/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_3/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_3/v*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_4/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_4/v
?
:Adam/graph_attention_sparse/kernel_4/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_4/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_4/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_4/v
?
8Adam/graph_attention_sparse/bias_4/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_4/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_4/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_4/v
?
DAdam/graph_attention_sparse/attn_kernel_self_4/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_4/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_4/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_4/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_4/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_4/v*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_5/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_5/v
?
:Adam/graph_attention_sparse/kernel_5/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_5/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_5/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_5/v
?
8Adam/graph_attention_sparse/bias_5/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_5/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_5/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_5/v
?
DAdam/graph_attention_sparse/attn_kernel_self_5/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_5/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_5/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_5/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_5/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_5/v*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_6/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_6/v
?
:Adam/graph_attention_sparse/kernel_6/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_6/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_6/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_6/v
?
8Adam/graph_attention_sparse/bias_6/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_6/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_6/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_6/v
?
DAdam/graph_attention_sparse/attn_kernel_self_6/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_6/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_6/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_6/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_6/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_6/v*
_output_shapes

:*
dtype0
?
&Adam/graph_attention_sparse/kernel_7/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/graph_attention_sparse/kernel_7/v
?
:Adam/graph_attention_sparse/kernel_7/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse/kernel_7/v*
_output_shapes
:	?*
dtype0
?
$Adam/graph_attention_sparse/bias_7/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/graph_attention_sparse/bias_7/v
?
8Adam/graph_attention_sparse/bias_7/v/Read/ReadVariableOpReadVariableOp$Adam/graph_attention_sparse/bias_7/v*
_output_shapes
:*
dtype0
?
0Adam/graph_attention_sparse/attn_kernel_self_7/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/graph_attention_sparse/attn_kernel_self_7/v
?
DAdam/graph_attention_sparse/attn_kernel_self_7/v/Read/ReadVariableOpReadVariableOp0Adam/graph_attention_sparse/attn_kernel_self_7/v*
_output_shapes

:*
dtype0
?
1Adam/graph_attention_sparse/attn_kernel_neigh_7/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/graph_attention_sparse/attn_kernel_neigh_7/v
?
EAdam/graph_attention_sparse/attn_kernel_neigh_7/v/Read/ReadVariableOpReadVariableOp1Adam/graph_attention_sparse/attn_kernel_neigh_7/v*
_output_shapes

:*
dtype0
?
(Adam/graph_attention_sparse_1/kernel_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/graph_attention_sparse_1/kernel_0/v
?
<Adam/graph_attention_sparse_1/kernel_0/v/Read/ReadVariableOpReadVariableOp(Adam/graph_attention_sparse_1/kernel_0/v*
_output_shapes

:@*
dtype0
?
&Adam/graph_attention_sparse_1/bias_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/graph_attention_sparse_1/bias_0/v
?
:Adam/graph_attention_sparse_1/bias_0/v/Read/ReadVariableOpReadVariableOp&Adam/graph_attention_sparse_1/bias_0/v*
_output_shapes
:*
dtype0
?
2Adam/graph_attention_sparse_1/attn_kernel_self_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/graph_attention_sparse_1/attn_kernel_self_0/v
?
FAdam/graph_attention_sparse_1/attn_kernel_self_0/v/Read/ReadVariableOpReadVariableOp2Adam/graph_attention_sparse_1/attn_kernel_self_0/v*
_output_shapes

:*
dtype0
?
3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53Adam/graph_attention_sparse_1/attn_kernel_neigh_0/v
?
GAdam/graph_attention_sparse_1/attn_kernel_neigh_0/v/Read/ReadVariableOpReadVariableOp3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/v*
_output_shapes

:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ʳ
value??B?? B??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer-9
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
 
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
kernels

biases
attn_kernels
ig_delta
	delta
ig_non_exist_edge
non_exist_edge
kernel_0

 bias_0
!attn_kernel_self_0
"attn_kernel_neigh_0
#kernel_1

$bias_1
%attn_kernel_self_1
&attn_kernel_neigh_1
'kernel_2

(bias_2
)attn_kernel_self_2
*attn_kernel_neigh_2
+kernel_3

,bias_3
-attn_kernel_self_3
.attn_kernel_neigh_3
/kernel_4

0bias_4
1attn_kernel_self_4
2attn_kernel_neigh_4
3kernel_5

4bias_5
5attn_kernel_self_5
6attn_kernel_neigh_5
7kernel_6

8bias_6
9attn_kernel_self_6
:attn_kernel_neigh_6
;kernel_7

<bias_7
=attn_kernel_self_7
>attn_kernel_neigh_7
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
R
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?
Gkernels

Hbiases
Iattn_kernels
Jig_delta
	Jdelta
Kig_non_exist_edge
Knon_exist_edge
Lkernel_0

Mbias_0
Nattn_kernel_self_0
Oattn_kernel_neigh_0
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
 
R
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
R
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
?
\iter

]beta_1

^beta_2
	_decay
`learning_ratem? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m?Lm?Mm?Nm?Om?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v?Lv?Mv?Nv?Ov?
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31
32
33
L34
M35
N36
O37
J38
K39
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31
L32
M33
N34
O35
 
?
	variables
trainable_variables
regularization_losses
alayer_metrics
bmetrics
cnon_trainable_variables
dlayer_regularization_losses

elayers
 
 
 
 
?
	variables
trainable_variables
regularization_losses
flayer_metrics
gmetrics
hnon_trainable_variables
ilayer_regularization_losses

jlayers
 
 
 
?
	variables
trainable_variables
regularization_losses
klayer_metrics
lmetrics
mnon_trainable_variables
nlayer_regularization_losses

olayers
8
0
#1
'2
+3
/4
35
76
;7
8
 0
$1
(2
,3
04
45
86
<7
8
p0
q1
r2
s3
t4
u5
v6
w7
mk
VARIABLE_VALUEgraph_attention_sparse/ig_delta8layer_with_weights-0/ig_delta/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE(graph_attention_sparse/ig_non_exist_edgeAlayer_with_weights-0/ig_non_exist_edge/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_08layer_with_weights-0/kernel_0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_06layer_with_weights-0/bias_0/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_0Blayer_with_weights-0/attn_kernel_self_0/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_0Clayer_with_weights-0/attn_kernel_neigh_0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_18layer_with_weights-0/kernel_1/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_16layer_with_weights-0/bias_1/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_1Blayer_with_weights-0/attn_kernel_self_1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_1Clayer_with_weights-0/attn_kernel_neigh_1/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_28layer_with_weights-0/kernel_2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_26layer_with_weights-0/bias_2/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_2Blayer_with_weights-0/attn_kernel_self_2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_2Clayer_with_weights-0/attn_kernel_neigh_2/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_38layer_with_weights-0/kernel_3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_36layer_with_weights-0/bias_3/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_3Blayer_with_weights-0/attn_kernel_self_3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_3Clayer_with_weights-0/attn_kernel_neigh_3/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_48layer_with_weights-0/kernel_4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_46layer_with_weights-0/bias_4/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_4Blayer_with_weights-0/attn_kernel_self_4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_4Clayer_with_weights-0/attn_kernel_neigh_4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_58layer_with_weights-0/kernel_5/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_56layer_with_weights-0/bias_5/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_5Blayer_with_weights-0/attn_kernel_self_5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_5Clayer_with_weights-0/attn_kernel_neigh_5/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_68layer_with_weights-0/kernel_6/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_66layer_with_weights-0/bias_6/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_6Blayer_with_weights-0/attn_kernel_self_6/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_6Clayer_with_weights-0/attn_kernel_neigh_6/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEgraph_attention_sparse/kernel_78layer_with_weights-0/kernel_7/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEgraph_attention_sparse/bias_76layer_with_weights-0/bias_7/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)graph_attention_sparse/attn_kernel_self_7Blayer_with_weights-0/attn_kernel_self_7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*graph_attention_sparse/attn_kernel_neigh_7Clayer_with_weights-0/attn_kernel_neigh_7/.ATTRIBUTES/VARIABLE_VALUE
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31
32
33
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31
 
?
?	variables
@trainable_variables
Aregularization_losses
xlayer_metrics
ymetrics
znon_trainable_variables
{layer_regularization_losses

|layers
 
 
 
?
C	variables
Dtrainable_variables
Eregularization_losses
}layer_metrics
~metrics
non_trainable_variables
 ?layer_regularization_losses
?layers

L0

M0

?0
om
VARIABLE_VALUE!graph_attention_sparse_1/ig_delta8layer_with_weights-1/ig_delta/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE*graph_attention_sparse_1/ig_non_exist_edgeAlayer_with_weights-1/ig_non_exist_edge/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE!graph_attention_sparse_1/kernel_08layer_with_weights-1/kernel_0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEgraph_attention_sparse_1/bias_06layer_with_weights-1/bias_0/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+graph_attention_sparse_1/attn_kernel_self_0Blayer_with_weights-1/attn_kernel_self_0/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,graph_attention_sparse_1/attn_kernel_neigh_0Clayer_with_weights-1/attn_kernel_neigh_0/.ATTRIBUTES/VARIABLE_VALUE
*
L0
M1
N2
O3
J4
K5

L0
M1
N2
O3
 
?
P	variables
Qtrainable_variables
Rregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
 
 
 
?
T	variables
Utrainable_variables
Vregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
 
 
 
?
X	variables
Ytrainable_variables
Zregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
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

?0
?1

0
1
J2
K3
 
N
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

!0
"1

%0
&1

)0
*1

-0
.1

10
21

50
61

90
:1

=0
>1
 
 

0
1
 
 
 
 
 
 
 

N0
O1
 
 

J0
K1
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

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_0/mTlayer_with_weights-0/kernel_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_0/mRlayer_with_weights-0/bias_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_0/m^layer_with_weights-0/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_0/m_layer_with_weights-0/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_1/mTlayer_with_weights-0/kernel_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_1/mRlayer_with_weights-0/bias_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_1/m^layer_with_weights-0/attn_kernel_self_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_1/m_layer_with_weights-0/attn_kernel_neigh_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_2/mTlayer_with_weights-0/kernel_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_2/mRlayer_with_weights-0/bias_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_2/m^layer_with_weights-0/attn_kernel_self_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_2/m_layer_with_weights-0/attn_kernel_neigh_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_3/mTlayer_with_weights-0/kernel_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_3/mRlayer_with_weights-0/bias_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_3/m^layer_with_weights-0/attn_kernel_self_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_3/m_layer_with_weights-0/attn_kernel_neigh_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_4/mTlayer_with_weights-0/kernel_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_4/mRlayer_with_weights-0/bias_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_4/m^layer_with_weights-0/attn_kernel_self_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_4/m_layer_with_weights-0/attn_kernel_neigh_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_5/mTlayer_with_weights-0/kernel_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_5/mRlayer_with_weights-0/bias_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_5/m^layer_with_weights-0/attn_kernel_self_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_5/m_layer_with_weights-0/attn_kernel_neigh_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_6/mTlayer_with_weights-0/kernel_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_6/mRlayer_with_weights-0/bias_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_6/m^layer_with_weights-0/attn_kernel_self_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_6/m_layer_with_weights-0/attn_kernel_neigh_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_7/mTlayer_with_weights-0/kernel_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_7/mRlayer_with_weights-0/bias_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_7/m^layer_with_weights-0/attn_kernel_self_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_7/m_layer_with_weights-0/attn_kernel_neigh_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/graph_attention_sparse_1/kernel_0/mTlayer_with_weights-1/kernel_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse_1/bias_0/mRlayer_with_weights-1/bias_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/graph_attention_sparse_1/attn_kernel_self_0/m^layer_with_weights-1/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/m_layer_with_weights-1/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_0/vTlayer_with_weights-0/kernel_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_0/vRlayer_with_weights-0/bias_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_0/v^layer_with_weights-0/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_0/v_layer_with_weights-0/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_1/vTlayer_with_weights-0/kernel_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_1/vRlayer_with_weights-0/bias_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_1/v^layer_with_weights-0/attn_kernel_self_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_1/v_layer_with_weights-0/attn_kernel_neigh_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_2/vTlayer_with_weights-0/kernel_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_2/vRlayer_with_weights-0/bias_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_2/v^layer_with_weights-0/attn_kernel_self_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_2/v_layer_with_weights-0/attn_kernel_neigh_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_3/vTlayer_with_weights-0/kernel_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_3/vRlayer_with_weights-0/bias_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_3/v^layer_with_weights-0/attn_kernel_self_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_3/v_layer_with_weights-0/attn_kernel_neigh_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_4/vTlayer_with_weights-0/kernel_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_4/vRlayer_with_weights-0/bias_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_4/v^layer_with_weights-0/attn_kernel_self_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_4/v_layer_with_weights-0/attn_kernel_neigh_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_5/vTlayer_with_weights-0/kernel_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_5/vRlayer_with_weights-0/bias_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_5/v^layer_with_weights-0/attn_kernel_self_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_5/v_layer_with_weights-0/attn_kernel_neigh_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_6/vTlayer_with_weights-0/kernel_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_6/vRlayer_with_weights-0/bias_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_6/v^layer_with_weights-0/attn_kernel_self_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_6/v_layer_with_weights-0/attn_kernel_neigh_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse/kernel_7/vTlayer_with_weights-0/kernel_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/graph_attention_sparse/bias_7/vRlayer_with_weights-0/bias_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/graph_attention_sparse/attn_kernel_self_7/v^layer_with_weights-0/attn_kernel_self_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/graph_attention_sparse/attn_kernel_neigh_7/v_layer_with_weights-0/attn_kernel_neigh_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/graph_attention_sparse_1/kernel_0/vTlayer_with_weights-1/kernel_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/graph_attention_sparse_1/bias_0/vRlayer_with_weights-1/bias_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/graph_attention_sparse_1/attn_kernel_self_0/v^layer_with_weights-1/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/v_layer_with_weights-1/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
t
serving_default_input_1Placeholder*$
_output_shapes
:??*
dtype0*
shape:??
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_input_3Placeholder*+
_output_shapes
:?????????*
dtype0	* 
shape:?????????
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4graph_attention_sparse/kernel_0)graph_attention_sparse/attn_kernel_self_0*graph_attention_sparse/attn_kernel_neigh_0graph_attention_sparse/bias_0graph_attention_sparse/kernel_1)graph_attention_sparse/attn_kernel_self_1*graph_attention_sparse/attn_kernel_neigh_1graph_attention_sparse/bias_1graph_attention_sparse/kernel_2)graph_attention_sparse/attn_kernel_self_2*graph_attention_sparse/attn_kernel_neigh_2graph_attention_sparse/bias_2graph_attention_sparse/kernel_3)graph_attention_sparse/attn_kernel_self_3*graph_attention_sparse/attn_kernel_neigh_3graph_attention_sparse/bias_3graph_attention_sparse/kernel_4)graph_attention_sparse/attn_kernel_self_4*graph_attention_sparse/attn_kernel_neigh_4graph_attention_sparse/bias_4graph_attention_sparse/kernel_5)graph_attention_sparse/attn_kernel_self_5*graph_attention_sparse/attn_kernel_neigh_5graph_attention_sparse/bias_5graph_attention_sparse/kernel_6)graph_attention_sparse/attn_kernel_self_6*graph_attention_sparse/attn_kernel_neigh_6graph_attention_sparse/bias_6graph_attention_sparse/kernel_7)graph_attention_sparse/attn_kernel_self_7*graph_attention_sparse/attn_kernel_neigh_7graph_attention_sparse/bias_7!graph_attention_sparse_1/kernel_0+graph_attention_sparse_1/attn_kernel_self_0,graph_attention_sparse_1/attn_kernel_neigh_0graph_attention_sparse_1/bias_0*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_10129
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?<
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3graph_attention_sparse/ig_delta/Read/ReadVariableOp<graph_attention_sparse/ig_non_exist_edge/Read/ReadVariableOp3graph_attention_sparse/kernel_0/Read/ReadVariableOp1graph_attention_sparse/bias_0/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_0/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_0/Read/ReadVariableOp3graph_attention_sparse/kernel_1/Read/ReadVariableOp1graph_attention_sparse/bias_1/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_1/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_1/Read/ReadVariableOp3graph_attention_sparse/kernel_2/Read/ReadVariableOp1graph_attention_sparse/bias_2/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_2/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_2/Read/ReadVariableOp3graph_attention_sparse/kernel_3/Read/ReadVariableOp1graph_attention_sparse/bias_3/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_3/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_3/Read/ReadVariableOp3graph_attention_sparse/kernel_4/Read/ReadVariableOp1graph_attention_sparse/bias_4/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_4/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_4/Read/ReadVariableOp3graph_attention_sparse/kernel_5/Read/ReadVariableOp1graph_attention_sparse/bias_5/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_5/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_5/Read/ReadVariableOp3graph_attention_sparse/kernel_6/Read/ReadVariableOp1graph_attention_sparse/bias_6/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_6/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_6/Read/ReadVariableOp3graph_attention_sparse/kernel_7/Read/ReadVariableOp1graph_attention_sparse/bias_7/Read/ReadVariableOp=graph_attention_sparse/attn_kernel_self_7/Read/ReadVariableOp>graph_attention_sparse/attn_kernel_neigh_7/Read/ReadVariableOp5graph_attention_sparse_1/ig_delta/Read/ReadVariableOp>graph_attention_sparse_1/ig_non_exist_edge/Read/ReadVariableOp5graph_attention_sparse_1/kernel_0/Read/ReadVariableOp3graph_attention_sparse_1/bias_0/Read/ReadVariableOp?graph_attention_sparse_1/attn_kernel_self_0/Read/ReadVariableOp@graph_attention_sparse_1/attn_kernel_neigh_0/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_0/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_0/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_0/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_0/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_1/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_1/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_1/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_1/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_2/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_2/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_2/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_2/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_3/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_3/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_3/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_3/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_4/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_4/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_4/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_4/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_5/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_5/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_5/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_5/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_6/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_6/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_6/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_6/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_7/m/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_7/m/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_7/m/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_7/m/Read/ReadVariableOp<Adam/graph_attention_sparse_1/kernel_0/m/Read/ReadVariableOp:Adam/graph_attention_sparse_1/bias_0/m/Read/ReadVariableOpFAdam/graph_attention_sparse_1/attn_kernel_self_0/m/Read/ReadVariableOpGAdam/graph_attention_sparse_1/attn_kernel_neigh_0/m/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_0/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_0/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_0/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_0/v/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_1/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_1/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_1/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_1/v/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_2/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_2/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_2/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_2/v/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_3/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_3/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_3/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_3/v/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_4/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_4/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_4/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_4/v/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_5/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_5/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_5/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_5/v/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_6/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_6/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_6/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_6/v/Read/ReadVariableOp:Adam/graph_attention_sparse/kernel_7/v/Read/ReadVariableOp8Adam/graph_attention_sparse/bias_7/v/Read/ReadVariableOpDAdam/graph_attention_sparse/attn_kernel_self_7/v/Read/ReadVariableOpEAdam/graph_attention_sparse/attn_kernel_neigh_7/v/Read/ReadVariableOp<Adam/graph_attention_sparse_1/kernel_0/v/Read/ReadVariableOp:Adam/graph_attention_sparse_1/bias_0/v/Read/ReadVariableOpFAdam/graph_attention_sparse_1/attn_kernel_self_0/v/Read/ReadVariableOpGAdam/graph_attention_sparse_1/attn_kernel_neigh_0/v/Read/ReadVariableOpConst*?
Tin
}2{	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_12593
?)
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegraph_attention_sparse/ig_delta(graph_attention_sparse/ig_non_exist_edgegraph_attention_sparse/kernel_0graph_attention_sparse/bias_0)graph_attention_sparse/attn_kernel_self_0*graph_attention_sparse/attn_kernel_neigh_0graph_attention_sparse/kernel_1graph_attention_sparse/bias_1)graph_attention_sparse/attn_kernel_self_1*graph_attention_sparse/attn_kernel_neigh_1graph_attention_sparse/kernel_2graph_attention_sparse/bias_2)graph_attention_sparse/attn_kernel_self_2*graph_attention_sparse/attn_kernel_neigh_2graph_attention_sparse/kernel_3graph_attention_sparse/bias_3)graph_attention_sparse/attn_kernel_self_3*graph_attention_sparse/attn_kernel_neigh_3graph_attention_sparse/kernel_4graph_attention_sparse/bias_4)graph_attention_sparse/attn_kernel_self_4*graph_attention_sparse/attn_kernel_neigh_4graph_attention_sparse/kernel_5graph_attention_sparse/bias_5)graph_attention_sparse/attn_kernel_self_5*graph_attention_sparse/attn_kernel_neigh_5graph_attention_sparse/kernel_6graph_attention_sparse/bias_6)graph_attention_sparse/attn_kernel_self_6*graph_attention_sparse/attn_kernel_neigh_6graph_attention_sparse/kernel_7graph_attention_sparse/bias_7)graph_attention_sparse/attn_kernel_self_7*graph_attention_sparse/attn_kernel_neigh_7!graph_attention_sparse_1/ig_delta*graph_attention_sparse_1/ig_non_exist_edge!graph_attention_sparse_1/kernel_0graph_attention_sparse_1/bias_0+graph_attention_sparse_1/attn_kernel_self_0,graph_attention_sparse_1/attn_kernel_neigh_0	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1&Adam/graph_attention_sparse/kernel_0/m$Adam/graph_attention_sparse/bias_0/m0Adam/graph_attention_sparse/attn_kernel_self_0/m1Adam/graph_attention_sparse/attn_kernel_neigh_0/m&Adam/graph_attention_sparse/kernel_1/m$Adam/graph_attention_sparse/bias_1/m0Adam/graph_attention_sparse/attn_kernel_self_1/m1Adam/graph_attention_sparse/attn_kernel_neigh_1/m&Adam/graph_attention_sparse/kernel_2/m$Adam/graph_attention_sparse/bias_2/m0Adam/graph_attention_sparse/attn_kernel_self_2/m1Adam/graph_attention_sparse/attn_kernel_neigh_2/m&Adam/graph_attention_sparse/kernel_3/m$Adam/graph_attention_sparse/bias_3/m0Adam/graph_attention_sparse/attn_kernel_self_3/m1Adam/graph_attention_sparse/attn_kernel_neigh_3/m&Adam/graph_attention_sparse/kernel_4/m$Adam/graph_attention_sparse/bias_4/m0Adam/graph_attention_sparse/attn_kernel_self_4/m1Adam/graph_attention_sparse/attn_kernel_neigh_4/m&Adam/graph_attention_sparse/kernel_5/m$Adam/graph_attention_sparse/bias_5/m0Adam/graph_attention_sparse/attn_kernel_self_5/m1Adam/graph_attention_sparse/attn_kernel_neigh_5/m&Adam/graph_attention_sparse/kernel_6/m$Adam/graph_attention_sparse/bias_6/m0Adam/graph_attention_sparse/attn_kernel_self_6/m1Adam/graph_attention_sparse/attn_kernel_neigh_6/m&Adam/graph_attention_sparse/kernel_7/m$Adam/graph_attention_sparse/bias_7/m0Adam/graph_attention_sparse/attn_kernel_self_7/m1Adam/graph_attention_sparse/attn_kernel_neigh_7/m(Adam/graph_attention_sparse_1/kernel_0/m&Adam/graph_attention_sparse_1/bias_0/m2Adam/graph_attention_sparse_1/attn_kernel_self_0/m3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/m&Adam/graph_attention_sparse/kernel_0/v$Adam/graph_attention_sparse/bias_0/v0Adam/graph_attention_sparse/attn_kernel_self_0/v1Adam/graph_attention_sparse/attn_kernel_neigh_0/v&Adam/graph_attention_sparse/kernel_1/v$Adam/graph_attention_sparse/bias_1/v0Adam/graph_attention_sparse/attn_kernel_self_1/v1Adam/graph_attention_sparse/attn_kernel_neigh_1/v&Adam/graph_attention_sparse/kernel_2/v$Adam/graph_attention_sparse/bias_2/v0Adam/graph_attention_sparse/attn_kernel_self_2/v1Adam/graph_attention_sparse/attn_kernel_neigh_2/v&Adam/graph_attention_sparse/kernel_3/v$Adam/graph_attention_sparse/bias_3/v0Adam/graph_attention_sparse/attn_kernel_self_3/v1Adam/graph_attention_sparse/attn_kernel_neigh_3/v&Adam/graph_attention_sparse/kernel_4/v$Adam/graph_attention_sparse/bias_4/v0Adam/graph_attention_sparse/attn_kernel_self_4/v1Adam/graph_attention_sparse/attn_kernel_neigh_4/v&Adam/graph_attention_sparse/kernel_5/v$Adam/graph_attention_sparse/bias_5/v0Adam/graph_attention_sparse/attn_kernel_self_5/v1Adam/graph_attention_sparse/attn_kernel_neigh_5/v&Adam/graph_attention_sparse/kernel_6/v$Adam/graph_attention_sparse/bias_6/v0Adam/graph_attention_sparse/attn_kernel_self_6/v1Adam/graph_attention_sparse/attn_kernel_neigh_6/v&Adam/graph_attention_sparse/kernel_7/v$Adam/graph_attention_sparse/bias_7/v0Adam/graph_attention_sparse/attn_kernel_self_7/v1Adam/graph_attention_sparse/attn_kernel_neigh_7/v(Adam/graph_attention_sparse_1/kernel_0/v&Adam/graph_attention_sparse_1/bias_0/v2Adam/graph_attention_sparse_1/attn_kernel_self_0/v3Adam/graph_attention_sparse_1/attn_kernel_neigh_0/v*?
Tin~
|2z*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_12966Α*
?	
?
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_11150
inputs_0	
inputs_1
identity	

identity_1

identity_2	p
SqueezeSqueezeinputs_0*
T0	*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezep
	Squeeze_1Squeezeinputs_1*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1?
SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor/dense_shaped
IdentityIdentitySqueeze:output:0*
T0	*'
_output_shapes
:?????????2

Identityf

Identity_1IdentitySqueeze_1:output:0*
T0*#
_output_shapes
:?????????2

Identity_1l

Identity_2Identity!SparseTensor/dense_shape:output:0*
T0	*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*=
_input_shapes,
*:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_9403

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consto
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
:?@2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?
  @   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
:?@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?@2
dropout/GreaterEqual{
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?@2
dropout/Castv
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*#
_output_shapes
:?@2
dropout/Mul_1a
IdentityIdentitydropout/Mul_1:z:0*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?
?
6__inference_graph_attention_sparse_layer_call_fn_12002
inputs_0

inputs	
inputs_1
inputs_2	
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*/
Tin(
&2$		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@*B
_read_only_resource_inputs$
" 	
 !"#*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_91802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
B
&__inference_lambda_layer_call_fn_12199

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_95962
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
?__inference_model_layer_call_and_return_conditional_losses_9796

inputs
inputs_1
inputs_2	
inputs_3
graph_attention_sparse_9718
graph_attention_sparse_9720
graph_attention_sparse_9722
graph_attention_sparse_9724
graph_attention_sparse_9726
graph_attention_sparse_9728
graph_attention_sparse_9730
graph_attention_sparse_9732
graph_attention_sparse_9734
graph_attention_sparse_9736
graph_attention_sparse_9738
graph_attention_sparse_9740
graph_attention_sparse_9742
graph_attention_sparse_9744
graph_attention_sparse_9746
graph_attention_sparse_9748
graph_attention_sparse_9750
graph_attention_sparse_9752
graph_attention_sparse_9754
graph_attention_sparse_9756
graph_attention_sparse_9758
graph_attention_sparse_9760
graph_attention_sparse_9762
graph_attention_sparse_9764
graph_attention_sparse_9766
graph_attention_sparse_9768
graph_attention_sparse_9770
graph_attention_sparse_9772
graph_attention_sparse_9774
graph_attention_sparse_9776
graph_attention_sparse_9778
graph_attention_sparse_9780!
graph_attention_sparse_1_9784!
graph_attention_sparse_1_9786!
graph_attention_sparse_1_9788!
graph_attention_sparse_1_9790
identity??dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?.graph_attention_sparse/StatefulPartitionedCall?0graph_attention_sparse_1/StatefulPartitionedCall?
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinputs_2inputs_3*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:?????????:?????????:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_84372,
*squeezed_sparse_conversion/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_84642!
dropout/StatefulPartitionedCall?
.graph_attention_sparse/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_9718graph_attention_sparse_9720graph_attention_sparse_9722graph_attention_sparse_9724graph_attention_sparse_9726graph_attention_sparse_9728graph_attention_sparse_9730graph_attention_sparse_9732graph_attention_sparse_9734graph_attention_sparse_9736graph_attention_sparse_9738graph_attention_sparse_9740graph_attention_sparse_9742graph_attention_sparse_9744graph_attention_sparse_9746graph_attention_sparse_9748graph_attention_sparse_9750graph_attention_sparse_9752graph_attention_sparse_9754graph_attention_sparse_9756graph_attention_sparse_9758graph_attention_sparse_9760graph_attention_sparse_9762graph_attention_sparse_9764graph_attention_sparse_9766graph_attention_sparse_9768graph_attention_sparse_9770graph_attention_sparse_9772graph_attention_sparse_9774graph_attention_sparse_9776graph_attention_sparse_9778graph_attention_sparse_9780*/
Tin(
&2$		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@*B
_read_only_resource_inputs$
" 	
 !"#*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_888720
.graph_attention_sparse/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall7graph_attention_sparse/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_94032#
!dropout_1/StatefulPartitionedCall?
0graph_attention_sparse_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_1_9784graph_attention_sparse_1_9786graph_attention_sparse_1_9788graph_attention_sparse_1_9790*
Tin

2		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_948422
0graph_attention_sparse_1/StatefulPartitionedCall?
gather_indices/PartitionedCallPartitionedCall9graph_attention_sparse_1/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gather_indices_layer_call_and_return_conditional_losses_95832 
gather_indices/PartitionedCall?
lambda/PartitionedCallPartitionedCall'gather_indices/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_95962
lambda/PartitionedCall?
IdentityIdentitylambda/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall/^graph_attention_sparse/StatefulPartitionedCall1^graph_attention_sparse_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2`
.graph_attention_sparse/StatefulPartitionedCall.graph_attention_sparse/StatefulPartitionedCall2d
0graph_attention_sparse_1/StatefulPartitionedCall0graph_attention_sparse_1/StatefulPartitionedCall:L H
$
_output_shapes
:??
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
8__inference_graph_attention_sparse_1_layer_call_fn_12157
inputs_0

inputs	
inputs_1
inputs_2	
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2*
Tin

2		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_94842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?@:?????????:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
\
@__inference_lambda_layer_call_and_return_conditional_losses_9596

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
?__inference_model_layer_call_and_return_conditional_losses_9614
input_1
input_2
input_3	
input_4
graph_attention_sparse_9326
graph_attention_sparse_9328
graph_attention_sparse_9330
graph_attention_sparse_9332
graph_attention_sparse_9334
graph_attention_sparse_9336
graph_attention_sparse_9338
graph_attention_sparse_9340
graph_attention_sparse_9342
graph_attention_sparse_9344
graph_attention_sparse_9346
graph_attention_sparse_9348
graph_attention_sparse_9350
graph_attention_sparse_9352
graph_attention_sparse_9354
graph_attention_sparse_9356
graph_attention_sparse_9358
graph_attention_sparse_9360
graph_attention_sparse_9362
graph_attention_sparse_9364
graph_attention_sparse_9366
graph_attention_sparse_9368
graph_attention_sparse_9370
graph_attention_sparse_9372
graph_attention_sparse_9374
graph_attention_sparse_9376
graph_attention_sparse_9378
graph_attention_sparse_9380
graph_attention_sparse_9382
graph_attention_sparse_9384
graph_attention_sparse_9386
graph_attention_sparse_9388!
graph_attention_sparse_1_9567!
graph_attention_sparse_1_9569!
graph_attention_sparse_1_9571!
graph_attention_sparse_1_9573
identity??dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?.graph_attention_sparse/StatefulPartitionedCall?0graph_attention_sparse_1/StatefulPartitionedCall?
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinput_3input_4*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:?????????:?????????:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_84372,
*squeezed_sparse_conversion/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_84642!
dropout/StatefulPartitionedCall?
.graph_attention_sparse/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_9326graph_attention_sparse_9328graph_attention_sparse_9330graph_attention_sparse_9332graph_attention_sparse_9334graph_attention_sparse_9336graph_attention_sparse_9338graph_attention_sparse_9340graph_attention_sparse_9342graph_attention_sparse_9344graph_attention_sparse_9346graph_attention_sparse_9348graph_attention_sparse_9350graph_attention_sparse_9352graph_attention_sparse_9354graph_attention_sparse_9356graph_attention_sparse_9358graph_attention_sparse_9360graph_attention_sparse_9362graph_attention_sparse_9364graph_attention_sparse_9366graph_attention_sparse_9368graph_attention_sparse_9370graph_attention_sparse_9372graph_attention_sparse_9374graph_attention_sparse_9376graph_attention_sparse_9378graph_attention_sparse_9380graph_attention_sparse_9382graph_attention_sparse_9384graph_attention_sparse_9386graph_attention_sparse_9388*/
Tin(
&2$		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@*B
_read_only_resource_inputs$
" 	
 !"#*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_888720
.graph_attention_sparse/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall7graph_attention_sparse/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_94032#
!dropout_1/StatefulPartitionedCall?
0graph_attention_sparse_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_1_9567graph_attention_sparse_1_9569graph_attention_sparse_1_9571graph_attention_sparse_1_9573*
Tin

2		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_948422
0graph_attention_sparse_1/StatefulPartitionedCall?
gather_indices/PartitionedCallPartitionedCall9graph_attention_sparse_1/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gather_indices_layer_call_and_return_conditional_losses_95832 
gather_indices/PartitionedCall?
lambda/PartitionedCallPartitionedCall'gather_indices/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_95962
lambda/PartitionedCall?
IdentityIdentitylambda/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall/^graph_attention_sparse/StatefulPartitionedCall1^graph_attention_sparse_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2`
.graph_attention_sparse/StatefulPartitionedCall.graph_attention_sparse/StatefulPartitionedCall2d
0graph_attention_sparse_1/StatefulPartitionedCall0graph_attention_sparse_1/StatefulPartitionedCall:M I
$
_output_shapes
:??
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?4
?
?__inference_model_layer_call_and_return_conditional_losses_9702
input_1
input_2
input_3	
input_4
graph_attention_sparse_9624
graph_attention_sparse_9626
graph_attention_sparse_9628
graph_attention_sparse_9630
graph_attention_sparse_9632
graph_attention_sparse_9634
graph_attention_sparse_9636
graph_attention_sparse_9638
graph_attention_sparse_9640
graph_attention_sparse_9642
graph_attention_sparse_9644
graph_attention_sparse_9646
graph_attention_sparse_9648
graph_attention_sparse_9650
graph_attention_sparse_9652
graph_attention_sparse_9654
graph_attention_sparse_9656
graph_attention_sparse_9658
graph_attention_sparse_9660
graph_attention_sparse_9662
graph_attention_sparse_9664
graph_attention_sparse_9666
graph_attention_sparse_9668
graph_attention_sparse_9670
graph_attention_sparse_9672
graph_attention_sparse_9674
graph_attention_sparse_9676
graph_attention_sparse_9678
graph_attention_sparse_9680
graph_attention_sparse_9682
graph_attention_sparse_9684
graph_attention_sparse_9686!
graph_attention_sparse_1_9690!
graph_attention_sparse_1_9692!
graph_attention_sparse_1_9694!
graph_attention_sparse_1_9696
identity??.graph_attention_sparse/StatefulPartitionedCall?0graph_attention_sparse_1/StatefulPartitionedCall?
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinput_3input_4*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:?????????:?????????:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_84372,
*squeezed_sparse_conversion/PartitionedCall?
dropout/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_84692
dropout/PartitionedCall?
.graph_attention_sparse/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_9624graph_attention_sparse_9626graph_attention_sparse_9628graph_attention_sparse_9630graph_attention_sparse_9632graph_attention_sparse_9634graph_attention_sparse_9636graph_attention_sparse_9638graph_attention_sparse_9640graph_attention_sparse_9642graph_attention_sparse_9644graph_attention_sparse_9646graph_attention_sparse_9648graph_attention_sparse_9650graph_attention_sparse_9652graph_attention_sparse_9654graph_attention_sparse_9656graph_attention_sparse_9658graph_attention_sparse_9660graph_attention_sparse_9662graph_attention_sparse_9664graph_attention_sparse_9666graph_attention_sparse_9668graph_attention_sparse_9670graph_attention_sparse_9672graph_attention_sparse_9674graph_attention_sparse_9676graph_attention_sparse_9678graph_attention_sparse_9680graph_attention_sparse_9682graph_attention_sparse_9684graph_attention_sparse_9686*/
Tin(
&2$		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@*B
_read_only_resource_inputs$
" 	
 !"#*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_918020
.graph_attention_sparse/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall7graph_attention_sparse/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_94082
dropout_1/PartitionedCall?
0graph_attention_sparse_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_1_9690graph_attention_sparse_1_9692graph_attention_sparse_1_9694graph_attention_sparse_1_9696*
Tin

2		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_953322
0graph_attention_sparse_1/StatefulPartitionedCall?
gather_indices/PartitionedCallPartitionedCall9graph_attention_sparse_1/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gather_indices_layer_call_and_return_conditional_losses_95832 
gather_indices/PartitionedCall?
lambda/PartitionedCallPartitionedCall'gather_indices/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_96002
lambda/PartitionedCall?
IdentityIdentitylambda/PartitionedCall:output:0/^graph_attention_sparse/StatefulPartitionedCall1^graph_attention_sparse_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::2`
.graph_attention_sparse/StatefulPartitionedCall.graph_attention_sparse/StatefulPartitionedCall2d
0graph_attention_sparse_1/StatefulPartitionedCall0graph_attention_sparse_1/StatefulPartitionedCall:M I
$
_output_shapes
:??
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?3
?
S__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_12141
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOph
SqueezeSqueezeinputs_0*
T0*
_output_shapes
:	?@*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelul
dropout/IdentityIdentityMatMul:product:0*
T0*
_output_shapes
:	?2
dropout/Identity?
dropout_1/IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_1/Identity?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputsdropout_1/Identity:output:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/Identity:output:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd_
stackPackBiasAdd:output:0*
N*
T0*#
_output_shapes
:?2
stackr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indiceso
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:	?2
MeanV
SoftmaxSoftmaxMean:output:0*
T0*
_output_shapes
:	?2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp*
T0*#
_output_shapes
:?2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?@:?????????:?????????:::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp:M I
#
_output_shapes
:?@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_11033
inputs_0
inputs_1
inputs_2	
inputs_3
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_97962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
?
?
%__inference_model_layer_call_fn_11113
inputs_0
inputs_1
inputs_2	
inputs_3
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_99642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
??
?
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_9180

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource%
!biasadd_1_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource$
 matmul_8_readvariableop_resource%
!biasadd_2_readvariableop_resource$
 matmul_9_readvariableop_resource%
!matmul_10_readvariableop_resource%
!matmul_11_readvariableop_resource%
!biasadd_3_readvariableop_resource%
!matmul_12_readvariableop_resource%
!matmul_13_readvariableop_resource%
!matmul_14_readvariableop_resource%
!biasadd_4_readvariableop_resource%
!matmul_15_readvariableop_resource%
!matmul_16_readvariableop_resource%
!matmul_17_readvariableop_resource%
!biasadd_5_readvariableop_resource%
!matmul_18_readvariableop_resource%
!matmul_19_readvariableop_resource%
!matmul_20_readvariableop_resource%
!biasadd_6_readvariableop_resource%
!matmul_21_readvariableop_resource%
!matmul_22_readvariableop_resource%
!matmul_23_readvariableop_resource%
!biasadd_7_readvariableop_resource
identity??BiasAdd/ReadVariableOp?BiasAdd_1/ReadVariableOp?BiasAdd_2/ReadVariableOp?BiasAdd_3/ReadVariableOp?BiasAdd_4/ReadVariableOp?BiasAdd_5/ReadVariableOp?BiasAdd_6/ReadVariableOp?BiasAdd_7/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_10/ReadVariableOp?MatMul_11/ReadVariableOp?MatMul_12/ReadVariableOp?MatMul_13/ReadVariableOp?MatMul_14/ReadVariableOp?MatMul_15/ReadVariableOp?MatMul_16/ReadVariableOp?MatMul_17/ReadVariableOp?MatMul_18/ReadVariableOp?MatMul_19/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_20/ReadVariableOp?MatMul_21/ReadVariableOp?MatMul_22/ReadVariableOp?MatMul_23/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?MatMul_8/ReadVariableOp?MatMul_9/ReadVariableOpg
SqueezeSqueezeinputs*
T0* 
_output_shapes
:
??*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelul
dropout/IdentityIdentityMatMul:product:0*
T0*
_output_shapes
:	?2
dropout/Identity?
dropout_1/IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_1/Identity?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputs_1dropout_1/Identity:output:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/Identity:output:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_3/ReadVariableOp{
MatMul_3MatMulSqueeze:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_3?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_4/ReadVariableOp}
MatMul_4MatMulMatMul_3:product:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp}
MatMul_5MatMulMatMul_3:product:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_5u
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_2/shapeu
	Reshape_2ReshapeMatMul_4:product:0Reshape_2/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_2:output:0strided_slice_2:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_2u
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_3/shapeu
	Reshape_3ReshapeMatMul_5:product:0Reshape_3/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputs_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_3:output:0strided_slice_3:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_3o
add_1AddV2GatherV2_2:output:0GatherV2_3:output:0*
T0*#
_output_shapes
:?????????2
add_1o
leaky_re_lu_1/LeakyRelu	LeakyRelu	add_1:z:0*#
_output_shapes
:?????????2
leaky_re_lu_1/LeakyRelur
dropout_2/IdentityIdentityMatMul_3:product:0*
T0*
_output_shapes
:	?2
dropout_2/Identity?
dropout_3/IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_3/Identity?
SparseTensor_3/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_3/dense_shape?
SparseSoftmax_1/SparseSoftmaxSparseSoftmaxinputs_1dropout_3/Identity:output:0#SparseTensor_3/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_1/SparseSoftmax?
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_1/SparseSoftmax:output:0#SparseTensor_3/dense_shape:output:0dropout_2/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMul?
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_1/ReadVariableOp?
	BiasAdd_1BiasAdd;SparseTensorDenseMatMul_1/SparseTensorDenseMatMul:product:0 BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_1?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_6/ReadVariableOp{
MatMul_6MatMulSqueeze:output:0MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_7/ReadVariableOp}
MatMul_7MatMulMatMul_6:product:0MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_7?
MatMul_8/ReadVariableOpReadVariableOp matmul_8_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_8/ReadVariableOp}
MatMul_8MatMulMatMul_6:product:0MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_8u
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_4/shapeu
	Reshape_4ReshapeMatMul_7:product:0Reshape_4/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_4
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputs_1strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_4:output:0strided_slice_4:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_4u
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_5/shapeu
	Reshape_5ReshapeMatMul_8:product:0Reshape_5/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_5
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceinputs_1strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_5:output:0strided_slice_5:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_5o
add_2AddV2GatherV2_4:output:0GatherV2_5:output:0*
T0*#
_output_shapes
:?????????2
add_2o
leaky_re_lu_2/LeakyRelu	LeakyRelu	add_2:z:0*#
_output_shapes
:?????????2
leaky_re_lu_2/LeakyRelur
dropout_4/IdentityIdentityMatMul_6:product:0*
T0*
_output_shapes
:	?2
dropout_4/Identity?
dropout_5/IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_5/Identity?
SparseTensor_4/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_4/dense_shape?
SparseSoftmax_2/SparseSoftmaxSparseSoftmaxinputs_1dropout_5/Identity:output:0#SparseTensor_4/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_2/SparseSoftmax?
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_2/SparseSoftmax:output:0#SparseTensor_4/dense_shape:output:0dropout_4/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMul?
BiasAdd_2/ReadVariableOpReadVariableOp!biasadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_2/ReadVariableOp?
	BiasAdd_2BiasAdd;SparseTensorDenseMatMul_2/SparseTensorDenseMatMul:product:0 BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_2?
MatMul_9/ReadVariableOpReadVariableOp matmul_9_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_9/ReadVariableOp{
MatMul_9MatMulSqueeze:output:0MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_9?
MatMul_10/ReadVariableOpReadVariableOp!matmul_10_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_10/ReadVariableOp?
	MatMul_10MatMulMatMul_9:product:0 MatMul_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_10?
MatMul_11/ReadVariableOpReadVariableOp!matmul_11_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_11/ReadVariableOp?
	MatMul_11MatMulMatMul_9:product:0 MatMul_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_11u
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_6/shapev
	Reshape_6ReshapeMatMul_10:product:0Reshape_6/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceinputs_1strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_6:output:0strided_slice_6:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_6u
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_7/shapev
	Reshape_7ReshapeMatMul_11:product:0Reshape_7/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceinputs_1strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_7:output:0strided_slice_7:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_7o
add_3AddV2GatherV2_6:output:0GatherV2_7:output:0*
T0*#
_output_shapes
:?????????2
add_3o
leaky_re_lu_3/LeakyRelu	LeakyRelu	add_3:z:0*#
_output_shapes
:?????????2
leaky_re_lu_3/LeakyRelur
dropout_6/IdentityIdentityMatMul_9:product:0*
T0*
_output_shapes
:	?2
dropout_6/Identity?
dropout_7/IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_7/Identity?
SparseTensor_5/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_5/dense_shape?
SparseSoftmax_3/SparseSoftmaxSparseSoftmaxinputs_1dropout_7/Identity:output:0#SparseTensor_5/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_3/SparseSoftmax?
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_3/SparseSoftmax:output:0#SparseTensor_5/dense_shape:output:0dropout_6/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMul?
BiasAdd_3/ReadVariableOpReadVariableOp!biasadd_3_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_3/ReadVariableOp?
	BiasAdd_3BiasAdd;SparseTensorDenseMatMul_3/SparseTensorDenseMatMul:product:0 BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_3?
MatMul_12/ReadVariableOpReadVariableOp!matmul_12_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_12/ReadVariableOp~
	MatMul_12MatMulSqueeze:output:0 MatMul_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_12?
MatMul_13/ReadVariableOpReadVariableOp!matmul_13_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_13/ReadVariableOp?
	MatMul_13MatMulMatMul_12:product:0 MatMul_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_13?
MatMul_14/ReadVariableOpReadVariableOp!matmul_14_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_14/ReadVariableOp?
	MatMul_14MatMulMatMul_12:product:0 MatMul_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_14u
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_8/shapev
	Reshape_8ReshapeMatMul_13:product:0Reshape_8/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceinputs_1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8d
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axis?

GatherV2_8GatherV2Reshape_8:output:0strided_slice_8:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_8u
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_9/shapev
	Reshape_9ReshapeMatMul_14:product:0Reshape_9/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_9
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceinputs_1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9d
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axis?

GatherV2_9GatherV2Reshape_9:output:0strided_slice_9:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_9o
add_4AddV2GatherV2_8:output:0GatherV2_9:output:0*
T0*#
_output_shapes
:?????????2
add_4o
leaky_re_lu_4/LeakyRelu	LeakyRelu	add_4:z:0*#
_output_shapes
:?????????2
leaky_re_lu_4/LeakyRelus
dropout_8/IdentityIdentityMatMul_12:product:0*
T0*
_output_shapes
:	?2
dropout_8/Identity?
dropout_9/IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_9/Identity?
SparseTensor_6/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_6/dense_shape?
SparseSoftmax_4/SparseSoftmaxSparseSoftmaxinputs_1dropout_9/Identity:output:0#SparseTensor_6/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_4/SparseSoftmax?
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_4/SparseSoftmax:output:0#SparseTensor_6/dense_shape:output:0dropout_8/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMul?
BiasAdd_4/ReadVariableOpReadVariableOp!biasadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_4/ReadVariableOp?
	BiasAdd_4BiasAdd;SparseTensorDenseMatMul_4/SparseTensorDenseMatMul:product:0 BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_4?
MatMul_15/ReadVariableOpReadVariableOp!matmul_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_15/ReadVariableOp~
	MatMul_15MatMulSqueeze:output:0 MatMul_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_15?
MatMul_16/ReadVariableOpReadVariableOp!matmul_16_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_16/ReadVariableOp?
	MatMul_16MatMulMatMul_15:product:0 MatMul_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_16?
MatMul_17/ReadVariableOpReadVariableOp!matmul_17_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_17/ReadVariableOp?
	MatMul_17MatMulMatMul_15:product:0 MatMul_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_17w
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_10/shapey

Reshape_10ReshapeMatMul_16:product:0Reshape_10/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_10?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceinputs_1strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10f
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axis?
GatherV2_10GatherV2Reshape_10:output:0strided_slice_10:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_10w
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_11/shapey

Reshape_11ReshapeMatMul_17:product:0Reshape_11/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_11?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceinputs_1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11f
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axis?
GatherV2_11GatherV2Reshape_11:output:0strided_slice_11:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_11q
add_5AddV2GatherV2_10:output:0GatherV2_11:output:0*
T0*#
_output_shapes
:?????????2
add_5o
leaky_re_lu_5/LeakyRelu	LeakyRelu	add_5:z:0*#
_output_shapes
:?????????2
leaky_re_lu_5/LeakyReluu
dropout_10/IdentityIdentityMatMul_15:product:0*
T0*
_output_shapes
:	?2
dropout_10/Identity?
dropout_11/IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_11/Identity?
SparseTensor_7/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_7/dense_shape?
SparseSoftmax_5/SparseSoftmaxSparseSoftmaxinputs_1dropout_11/Identity:output:0#SparseTensor_7/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_5/SparseSoftmax?
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_5/SparseSoftmax:output:0#SparseTensor_7/dense_shape:output:0dropout_10/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMul?
BiasAdd_5/ReadVariableOpReadVariableOp!biasadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_5/ReadVariableOp?
	BiasAdd_5BiasAdd;SparseTensorDenseMatMul_5/SparseTensorDenseMatMul:product:0 BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_5?
MatMul_18/ReadVariableOpReadVariableOp!matmul_18_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_18/ReadVariableOp~
	MatMul_18MatMulSqueeze:output:0 MatMul_18/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_18?
MatMul_19/ReadVariableOpReadVariableOp!matmul_19_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_19/ReadVariableOp?
	MatMul_19MatMulMatMul_18:product:0 MatMul_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_19?
MatMul_20/ReadVariableOpReadVariableOp!matmul_20_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_20/ReadVariableOp?
	MatMul_20MatMulMatMul_18:product:0 MatMul_20/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_20w
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_12/shapey

Reshape_12ReshapeMatMul_19:product:0Reshape_12/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_12?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSliceinputs_1strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12f
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axis?
GatherV2_12GatherV2Reshape_12:output:0strided_slice_12:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_12w
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_13/shapey

Reshape_13ReshapeMatMul_20:product:0Reshape_13/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_13?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSliceinputs_1strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13f
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axis?
GatherV2_13GatherV2Reshape_13:output:0strided_slice_13:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_13q
add_6AddV2GatherV2_12:output:0GatherV2_13:output:0*
T0*#
_output_shapes
:?????????2
add_6o
leaky_re_lu_6/LeakyRelu	LeakyRelu	add_6:z:0*#
_output_shapes
:?????????2
leaky_re_lu_6/LeakyReluu
dropout_12/IdentityIdentityMatMul_18:product:0*
T0*
_output_shapes
:	?2
dropout_12/Identity?
dropout_13/IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_13/Identity?
SparseTensor_8/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_8/dense_shape?
SparseSoftmax_6/SparseSoftmaxSparseSoftmaxinputs_1dropout_13/Identity:output:0#SparseTensor_8/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_6/SparseSoftmax?
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_6/SparseSoftmax:output:0#SparseTensor_8/dense_shape:output:0dropout_12/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMul?
BiasAdd_6/ReadVariableOpReadVariableOp!biasadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_6/ReadVariableOp?
	BiasAdd_6BiasAdd;SparseTensorDenseMatMul_6/SparseTensorDenseMatMul:product:0 BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_6?
MatMul_21/ReadVariableOpReadVariableOp!matmul_21_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_21/ReadVariableOp~
	MatMul_21MatMulSqueeze:output:0 MatMul_21/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_21?
MatMul_22/ReadVariableOpReadVariableOp!matmul_22_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_22/ReadVariableOp?
	MatMul_22MatMulMatMul_21:product:0 MatMul_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_22?
MatMul_23/ReadVariableOpReadVariableOp!matmul_23_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_23/ReadVariableOp?
	MatMul_23MatMulMatMul_21:product:0 MatMul_23/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_23w
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_14/shapey

Reshape_14ReshapeMatMul_22:product:0Reshape_14/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_14?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSliceinputs_1strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14f
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axis?
GatherV2_14GatherV2Reshape_14:output:0strided_slice_14:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_14w
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_15/shapey

Reshape_15ReshapeMatMul_23:product:0Reshape_15/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_15?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSliceinputs_1strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15f
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axis?
GatherV2_15GatherV2Reshape_15:output:0strided_slice_15:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_15q
add_7AddV2GatherV2_14:output:0GatherV2_15:output:0*
T0*#
_output_shapes
:?????????2
add_7o
leaky_re_lu_7/LeakyRelu	LeakyRelu	add_7:z:0*#
_output_shapes
:?????????2
leaky_re_lu_7/LeakyReluu
dropout_14/IdentityIdentityMatMul_21:product:0*
T0*
_output_shapes
:	?2
dropout_14/Identity?
dropout_15/IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_15/Identity?
SparseTensor_9/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_9/dense_shape?
SparseSoftmax_7/SparseSoftmaxSparseSoftmaxinputs_1dropout_15/Identity:output:0#SparseTensor_9/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_7/SparseSoftmax?
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_7/SparseSoftmax:output:0#SparseTensor_9/dense_shape:output:0dropout_14/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMul?
BiasAdd_7/ReadVariableOpReadVariableOp!biasadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_7/ReadVariableOp?
	BiasAdd_7BiasAdd;SparseTensorDenseMatMul_7/SparseTensorDenseMatMul:product:0 BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_7\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2BiasAdd:output:0BiasAdd_1:output:0BiasAdd_2:output:0BiasAdd_3:output:0BiasAdd_4:output:0BiasAdd_5:output:0BiasAdd_6:output:0BiasAdd_7:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	?@2
concatL
EluEluconcat:output:0*
T0*
_output_shapes
:	?@2
Elub
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsElu:activations:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?@2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^BiasAdd_2/ReadVariableOp^BiasAdd_3/ReadVariableOp^BiasAdd_4/ReadVariableOp^BiasAdd_5/ReadVariableOp^BiasAdd_6/ReadVariableOp^BiasAdd_7/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_10/ReadVariableOp^MatMul_11/ReadVariableOp^MatMul_12/ReadVariableOp^MatMul_13/ReadVariableOp^MatMul_14/ReadVariableOp^MatMul_15/ReadVariableOp^MatMul_16/ReadVariableOp^MatMul_17/ReadVariableOp^MatMul_18/ReadVariableOp^MatMul_19/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_20/ReadVariableOp^MatMul_21/ReadVariableOp^MatMul_22/ReadVariableOp^MatMul_23/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:::::::::::::::::::::::::::::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
BiasAdd_2/ReadVariableOpBiasAdd_2/ReadVariableOp24
BiasAdd_3/ReadVariableOpBiasAdd_3/ReadVariableOp24
BiasAdd_4/ReadVariableOpBiasAdd_4/ReadVariableOp24
BiasAdd_5/ReadVariableOpBiasAdd_5/ReadVariableOp24
BiasAdd_6/ReadVariableOpBiasAdd_6/ReadVariableOp24
BiasAdd_7/ReadVariableOpBiasAdd_7/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp24
MatMul_10/ReadVariableOpMatMul_10/ReadVariableOp24
MatMul_11/ReadVariableOpMatMul_11/ReadVariableOp24
MatMul_12/ReadVariableOpMatMul_12/ReadVariableOp24
MatMul_13/ReadVariableOpMatMul_13/ReadVariableOp24
MatMul_14/ReadVariableOpMatMul_14/ReadVariableOp24
MatMul_15/ReadVariableOpMatMul_15/ReadVariableOp24
MatMul_16/ReadVariableOpMatMul_16/ReadVariableOp24
MatMul_17/ReadVariableOpMatMul_17/ReadVariableOp24
MatMul_18/ReadVariableOpMatMul_18/ReadVariableOp24
MatMul_19/ReadVariableOpMatMul_19/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp24
MatMul_20/ReadVariableOpMatMul_20/ReadVariableOp24
MatMul_21/ReadVariableOpMatMul_21/ReadVariableOp24
MatMul_22/ReadVariableOpMatMul_22/ReadVariableOp24
MatMul_23/ReadVariableOpMatMul_23/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp22
MatMul_8/ReadVariableOpMatMul_8/ReadVariableOp22
MatMul_9/ReadVariableOpMatMul_9/ReadVariableOp:L H
$
_output_shapes
:??
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
??
?
Q__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_11858
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource%
!biasadd_1_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource$
 matmul_8_readvariableop_resource%
!biasadd_2_readvariableop_resource$
 matmul_9_readvariableop_resource%
!matmul_10_readvariableop_resource%
!matmul_11_readvariableop_resource%
!biasadd_3_readvariableop_resource%
!matmul_12_readvariableop_resource%
!matmul_13_readvariableop_resource%
!matmul_14_readvariableop_resource%
!biasadd_4_readvariableop_resource%
!matmul_15_readvariableop_resource%
!matmul_16_readvariableop_resource%
!matmul_17_readvariableop_resource%
!biasadd_5_readvariableop_resource%
!matmul_18_readvariableop_resource%
!matmul_19_readvariableop_resource%
!matmul_20_readvariableop_resource%
!biasadd_6_readvariableop_resource%
!matmul_21_readvariableop_resource%
!matmul_22_readvariableop_resource%
!matmul_23_readvariableop_resource%
!biasadd_7_readvariableop_resource
identity??BiasAdd/ReadVariableOp?BiasAdd_1/ReadVariableOp?BiasAdd_2/ReadVariableOp?BiasAdd_3/ReadVariableOp?BiasAdd_4/ReadVariableOp?BiasAdd_5/ReadVariableOp?BiasAdd_6/ReadVariableOp?BiasAdd_7/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_10/ReadVariableOp?MatMul_11/ReadVariableOp?MatMul_12/ReadVariableOp?MatMul_13/ReadVariableOp?MatMul_14/ReadVariableOp?MatMul_15/ReadVariableOp?MatMul_16/ReadVariableOp?MatMul_17/ReadVariableOp?MatMul_18/ReadVariableOp?MatMul_19/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_20/ReadVariableOp?MatMul_21/ReadVariableOp?MatMul_22/ReadVariableOp?MatMul_23/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?MatMul_8/ReadVariableOp?MatMul_9/ReadVariableOpi
SqueezeSqueezeinputs_0*
T0* 
_output_shapes
:
??*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelul
dropout/IdentityIdentityMatMul:product:0*
T0*
_output_shapes
:	?2
dropout/Identity?
dropout_1/IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_1/Identity?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputsdropout_1/Identity:output:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/Identity:output:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_3/ReadVariableOp{
MatMul_3MatMulSqueeze:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_3?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_4/ReadVariableOp}
MatMul_4MatMulMatMul_3:product:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp}
MatMul_5MatMulMatMul_3:product:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_5u
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_2/shapeu
	Reshape_2ReshapeMatMul_4:product:0Reshape_2/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_2:output:0strided_slice_2:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_2u
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_3/shapeu
	Reshape_3ReshapeMatMul_5:product:0Reshape_3/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_3:output:0strided_slice_3:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_3o
add_1AddV2GatherV2_2:output:0GatherV2_3:output:0*
T0*#
_output_shapes
:?????????2
add_1o
leaky_re_lu_1/LeakyRelu	LeakyRelu	add_1:z:0*#
_output_shapes
:?????????2
leaky_re_lu_1/LeakyRelur
dropout_2/IdentityIdentityMatMul_3:product:0*
T0*
_output_shapes
:	?2
dropout_2/Identity?
dropout_3/IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_3/Identity?
SparseTensor_3/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_3/dense_shape?
SparseSoftmax_1/SparseSoftmaxSparseSoftmaxinputsdropout_3/Identity:output:0#SparseTensor_3/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_1/SparseSoftmax?
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_1/SparseSoftmax:output:0#SparseTensor_3/dense_shape:output:0dropout_2/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMul?
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_1/ReadVariableOp?
	BiasAdd_1BiasAdd;SparseTensorDenseMatMul_1/SparseTensorDenseMatMul:product:0 BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_1?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_6/ReadVariableOp{
MatMul_6MatMulSqueeze:output:0MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_7/ReadVariableOp}
MatMul_7MatMulMatMul_6:product:0MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_7?
MatMul_8/ReadVariableOpReadVariableOp matmul_8_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_8/ReadVariableOp}
MatMul_8MatMulMatMul_6:product:0MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_8u
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_4/shapeu
	Reshape_4ReshapeMatMul_7:product:0Reshape_4/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_4
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_4:output:0strided_slice_4:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_4u
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_5/shapeu
	Reshape_5ReshapeMatMul_8:product:0Reshape_5/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_5
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_5:output:0strided_slice_5:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_5o
add_2AddV2GatherV2_4:output:0GatherV2_5:output:0*
T0*#
_output_shapes
:?????????2
add_2o
leaky_re_lu_2/LeakyRelu	LeakyRelu	add_2:z:0*#
_output_shapes
:?????????2
leaky_re_lu_2/LeakyRelur
dropout_4/IdentityIdentityMatMul_6:product:0*
T0*
_output_shapes
:	?2
dropout_4/Identity?
dropout_5/IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_5/Identity?
SparseTensor_4/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_4/dense_shape?
SparseSoftmax_2/SparseSoftmaxSparseSoftmaxinputsdropout_5/Identity:output:0#SparseTensor_4/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_2/SparseSoftmax?
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_2/SparseSoftmax:output:0#SparseTensor_4/dense_shape:output:0dropout_4/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMul?
BiasAdd_2/ReadVariableOpReadVariableOp!biasadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_2/ReadVariableOp?
	BiasAdd_2BiasAdd;SparseTensorDenseMatMul_2/SparseTensorDenseMatMul:product:0 BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_2?
MatMul_9/ReadVariableOpReadVariableOp matmul_9_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_9/ReadVariableOp{
MatMul_9MatMulSqueeze:output:0MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_9?
MatMul_10/ReadVariableOpReadVariableOp!matmul_10_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_10/ReadVariableOp?
	MatMul_10MatMulMatMul_9:product:0 MatMul_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_10?
MatMul_11/ReadVariableOpReadVariableOp!matmul_11_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_11/ReadVariableOp?
	MatMul_11MatMulMatMul_9:product:0 MatMul_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_11u
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_6/shapev
	Reshape_6ReshapeMatMul_10:product:0Reshape_6/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_6:output:0strided_slice_6:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_6u
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_7/shapev
	Reshape_7ReshapeMatMul_11:product:0Reshape_7/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceinputsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_7:output:0strided_slice_7:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_7o
add_3AddV2GatherV2_6:output:0GatherV2_7:output:0*
T0*#
_output_shapes
:?????????2
add_3o
leaky_re_lu_3/LeakyRelu	LeakyRelu	add_3:z:0*#
_output_shapes
:?????????2
leaky_re_lu_3/LeakyRelur
dropout_6/IdentityIdentityMatMul_9:product:0*
T0*
_output_shapes
:	?2
dropout_6/Identity?
dropout_7/IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_7/Identity?
SparseTensor_5/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_5/dense_shape?
SparseSoftmax_3/SparseSoftmaxSparseSoftmaxinputsdropout_7/Identity:output:0#SparseTensor_5/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_3/SparseSoftmax?
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_3/SparseSoftmax:output:0#SparseTensor_5/dense_shape:output:0dropout_6/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMul?
BiasAdd_3/ReadVariableOpReadVariableOp!biasadd_3_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_3/ReadVariableOp?
	BiasAdd_3BiasAdd;SparseTensorDenseMatMul_3/SparseTensorDenseMatMul:product:0 BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_3?
MatMul_12/ReadVariableOpReadVariableOp!matmul_12_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_12/ReadVariableOp~
	MatMul_12MatMulSqueeze:output:0 MatMul_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_12?
MatMul_13/ReadVariableOpReadVariableOp!matmul_13_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_13/ReadVariableOp?
	MatMul_13MatMulMatMul_12:product:0 MatMul_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_13?
MatMul_14/ReadVariableOpReadVariableOp!matmul_14_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_14/ReadVariableOp?
	MatMul_14MatMulMatMul_12:product:0 MatMul_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_14u
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_8/shapev
	Reshape_8ReshapeMatMul_13:product:0Reshape_8/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceinputsstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8d
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axis?

GatherV2_8GatherV2Reshape_8:output:0strided_slice_8:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_8u
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_9/shapev
	Reshape_9ReshapeMatMul_14:product:0Reshape_9/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_9
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceinputsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9d
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axis?

GatherV2_9GatherV2Reshape_9:output:0strided_slice_9:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_9o
add_4AddV2GatherV2_8:output:0GatherV2_9:output:0*
T0*#
_output_shapes
:?????????2
add_4o
leaky_re_lu_4/LeakyRelu	LeakyRelu	add_4:z:0*#
_output_shapes
:?????????2
leaky_re_lu_4/LeakyRelus
dropout_8/IdentityIdentityMatMul_12:product:0*
T0*
_output_shapes
:	?2
dropout_8/Identity?
dropout_9/IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_9/Identity?
SparseTensor_6/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_6/dense_shape?
SparseSoftmax_4/SparseSoftmaxSparseSoftmaxinputsdropout_9/Identity:output:0#SparseTensor_6/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_4/SparseSoftmax?
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_4/SparseSoftmax:output:0#SparseTensor_6/dense_shape:output:0dropout_8/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMul?
BiasAdd_4/ReadVariableOpReadVariableOp!biasadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_4/ReadVariableOp?
	BiasAdd_4BiasAdd;SparseTensorDenseMatMul_4/SparseTensorDenseMatMul:product:0 BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_4?
MatMul_15/ReadVariableOpReadVariableOp!matmul_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_15/ReadVariableOp~
	MatMul_15MatMulSqueeze:output:0 MatMul_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_15?
MatMul_16/ReadVariableOpReadVariableOp!matmul_16_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_16/ReadVariableOp?
	MatMul_16MatMulMatMul_15:product:0 MatMul_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_16?
MatMul_17/ReadVariableOpReadVariableOp!matmul_17_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_17/ReadVariableOp?
	MatMul_17MatMulMatMul_15:product:0 MatMul_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_17w
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_10/shapey

Reshape_10ReshapeMatMul_16:product:0Reshape_10/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_10?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceinputsstrided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10f
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axis?
GatherV2_10GatherV2Reshape_10:output:0strided_slice_10:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_10w
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_11/shapey

Reshape_11ReshapeMatMul_17:product:0Reshape_11/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_11?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceinputsstrided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11f
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axis?
GatherV2_11GatherV2Reshape_11:output:0strided_slice_11:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_11q
add_5AddV2GatherV2_10:output:0GatherV2_11:output:0*
T0*#
_output_shapes
:?????????2
add_5o
leaky_re_lu_5/LeakyRelu	LeakyRelu	add_5:z:0*#
_output_shapes
:?????????2
leaky_re_lu_5/LeakyReluu
dropout_10/IdentityIdentityMatMul_15:product:0*
T0*
_output_shapes
:	?2
dropout_10/Identity?
dropout_11/IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_11/Identity?
SparseTensor_7/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_7/dense_shape?
SparseSoftmax_5/SparseSoftmaxSparseSoftmaxinputsdropout_11/Identity:output:0#SparseTensor_7/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_5/SparseSoftmax?
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_5/SparseSoftmax:output:0#SparseTensor_7/dense_shape:output:0dropout_10/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMul?
BiasAdd_5/ReadVariableOpReadVariableOp!biasadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_5/ReadVariableOp?
	BiasAdd_5BiasAdd;SparseTensorDenseMatMul_5/SparseTensorDenseMatMul:product:0 BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_5?
MatMul_18/ReadVariableOpReadVariableOp!matmul_18_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_18/ReadVariableOp~
	MatMul_18MatMulSqueeze:output:0 MatMul_18/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_18?
MatMul_19/ReadVariableOpReadVariableOp!matmul_19_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_19/ReadVariableOp?
	MatMul_19MatMulMatMul_18:product:0 MatMul_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_19?
MatMul_20/ReadVariableOpReadVariableOp!matmul_20_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_20/ReadVariableOp?
	MatMul_20MatMulMatMul_18:product:0 MatMul_20/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_20w
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_12/shapey

Reshape_12ReshapeMatMul_19:product:0Reshape_12/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_12?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSliceinputsstrided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12f
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axis?
GatherV2_12GatherV2Reshape_12:output:0strided_slice_12:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_12w
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_13/shapey

Reshape_13ReshapeMatMul_20:product:0Reshape_13/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_13?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSliceinputsstrided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13f
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axis?
GatherV2_13GatherV2Reshape_13:output:0strided_slice_13:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_13q
add_6AddV2GatherV2_12:output:0GatherV2_13:output:0*
T0*#
_output_shapes
:?????????2
add_6o
leaky_re_lu_6/LeakyRelu	LeakyRelu	add_6:z:0*#
_output_shapes
:?????????2
leaky_re_lu_6/LeakyReluu
dropout_12/IdentityIdentityMatMul_18:product:0*
T0*
_output_shapes
:	?2
dropout_12/Identity?
dropout_13/IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_13/Identity?
SparseTensor_8/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_8/dense_shape?
SparseSoftmax_6/SparseSoftmaxSparseSoftmaxinputsdropout_13/Identity:output:0#SparseTensor_8/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_6/SparseSoftmax?
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_6/SparseSoftmax:output:0#SparseTensor_8/dense_shape:output:0dropout_12/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMul?
BiasAdd_6/ReadVariableOpReadVariableOp!biasadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_6/ReadVariableOp?
	BiasAdd_6BiasAdd;SparseTensorDenseMatMul_6/SparseTensorDenseMatMul:product:0 BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_6?
MatMul_21/ReadVariableOpReadVariableOp!matmul_21_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_21/ReadVariableOp~
	MatMul_21MatMulSqueeze:output:0 MatMul_21/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_21?
MatMul_22/ReadVariableOpReadVariableOp!matmul_22_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_22/ReadVariableOp?
	MatMul_22MatMulMatMul_21:product:0 MatMul_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_22?
MatMul_23/ReadVariableOpReadVariableOp!matmul_23_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_23/ReadVariableOp?
	MatMul_23MatMulMatMul_21:product:0 MatMul_23/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_23w
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_14/shapey

Reshape_14ReshapeMatMul_22:product:0Reshape_14/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_14?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSliceinputsstrided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14f
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axis?
GatherV2_14GatherV2Reshape_14:output:0strided_slice_14:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_14w
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_15/shapey

Reshape_15ReshapeMatMul_23:product:0Reshape_15/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_15?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSliceinputsstrided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15f
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axis?
GatherV2_15GatherV2Reshape_15:output:0strided_slice_15:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_15q
add_7AddV2GatherV2_14:output:0GatherV2_15:output:0*
T0*#
_output_shapes
:?????????2
add_7o
leaky_re_lu_7/LeakyRelu	LeakyRelu	add_7:z:0*#
_output_shapes
:?????????2
leaky_re_lu_7/LeakyReluu
dropout_14/IdentityIdentityMatMul_21:product:0*
T0*
_output_shapes
:	?2
dropout_14/Identity?
dropout_15/IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_15/Identity?
SparseTensor_9/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_9/dense_shape?
SparseSoftmax_7/SparseSoftmaxSparseSoftmaxinputsdropout_15/Identity:output:0#SparseTensor_9/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_7/SparseSoftmax?
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_7/SparseSoftmax:output:0#SparseTensor_9/dense_shape:output:0dropout_14/Identity:output:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMul?
BiasAdd_7/ReadVariableOpReadVariableOp!biasadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_7/ReadVariableOp?
	BiasAdd_7BiasAdd;SparseTensorDenseMatMul_7/SparseTensorDenseMatMul:product:0 BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_7\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2BiasAdd:output:0BiasAdd_1:output:0BiasAdd_2:output:0BiasAdd_3:output:0BiasAdd_4:output:0BiasAdd_5:output:0BiasAdd_6:output:0BiasAdd_7:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	?@2
concatL
EluEluconcat:output:0*
T0*
_output_shapes
:	?@2
Elub
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsElu:activations:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?@2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^BiasAdd_2/ReadVariableOp^BiasAdd_3/ReadVariableOp^BiasAdd_4/ReadVariableOp^BiasAdd_5/ReadVariableOp^BiasAdd_6/ReadVariableOp^BiasAdd_7/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_10/ReadVariableOp^MatMul_11/ReadVariableOp^MatMul_12/ReadVariableOp^MatMul_13/ReadVariableOp^MatMul_14/ReadVariableOp^MatMul_15/ReadVariableOp^MatMul_16/ReadVariableOp^MatMul_17/ReadVariableOp^MatMul_18/ReadVariableOp^MatMul_19/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_20/ReadVariableOp^MatMul_21/ReadVariableOp^MatMul_22/ReadVariableOp^MatMul_23/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:::::::::::::::::::::::::::::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
BiasAdd_2/ReadVariableOpBiasAdd_2/ReadVariableOp24
BiasAdd_3/ReadVariableOpBiasAdd_3/ReadVariableOp24
BiasAdd_4/ReadVariableOpBiasAdd_4/ReadVariableOp24
BiasAdd_5/ReadVariableOpBiasAdd_5/ReadVariableOp24
BiasAdd_6/ReadVariableOpBiasAdd_6/ReadVariableOp24
BiasAdd_7/ReadVariableOpBiasAdd_7/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp24
MatMul_10/ReadVariableOpMatMul_10/ReadVariableOp24
MatMul_11/ReadVariableOpMatMul_11/ReadVariableOp24
MatMul_12/ReadVariableOpMatMul_12/ReadVariableOp24
MatMul_13/ReadVariableOpMatMul_13/ReadVariableOp24
MatMul_14/ReadVariableOpMatMul_14/ReadVariableOp24
MatMul_15/ReadVariableOpMatMul_15/ReadVariableOp24
MatMul_16/ReadVariableOpMatMul_16/ReadVariableOp24
MatMul_17/ReadVariableOpMatMul_17/ReadVariableOp24
MatMul_18/ReadVariableOpMatMul_18/ReadVariableOp24
MatMul_19/ReadVariableOpMatMul_19/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp24
MatMul_20/ReadVariableOpMatMul_20/ReadVariableOp24
MatMul_21/ReadVariableOpMatMul_21/ReadVariableOp24
MatMul_22/ReadVariableOpMatMul_22/ReadVariableOp24
MatMul_23/ReadVariableOpMatMul_23/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp22
MatMul_8/ReadVariableOpMatMul_8/ReadVariableOp22
MatMul_9/ReadVariableOpMatMul_9/ReadVariableOp:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
??
? 
@__inference_model_layer_call_and_return_conditional_losses_10953
inputs_0
inputs_1
inputs_2	
inputs_39
5graph_attention_sparse_matmul_readvariableop_resource;
7graph_attention_sparse_matmul_1_readvariableop_resource;
7graph_attention_sparse_matmul_2_readvariableop_resource:
6graph_attention_sparse_biasadd_readvariableop_resource;
7graph_attention_sparse_matmul_3_readvariableop_resource;
7graph_attention_sparse_matmul_4_readvariableop_resource;
7graph_attention_sparse_matmul_5_readvariableop_resource<
8graph_attention_sparse_biasadd_1_readvariableop_resource;
7graph_attention_sparse_matmul_6_readvariableop_resource;
7graph_attention_sparse_matmul_7_readvariableop_resource;
7graph_attention_sparse_matmul_8_readvariableop_resource<
8graph_attention_sparse_biasadd_2_readvariableop_resource;
7graph_attention_sparse_matmul_9_readvariableop_resource<
8graph_attention_sparse_matmul_10_readvariableop_resource<
8graph_attention_sparse_matmul_11_readvariableop_resource<
8graph_attention_sparse_biasadd_3_readvariableop_resource<
8graph_attention_sparse_matmul_12_readvariableop_resource<
8graph_attention_sparse_matmul_13_readvariableop_resource<
8graph_attention_sparse_matmul_14_readvariableop_resource<
8graph_attention_sparse_biasadd_4_readvariableop_resource<
8graph_attention_sparse_matmul_15_readvariableop_resource<
8graph_attention_sparse_matmul_16_readvariableop_resource<
8graph_attention_sparse_matmul_17_readvariableop_resource<
8graph_attention_sparse_biasadd_5_readvariableop_resource<
8graph_attention_sparse_matmul_18_readvariableop_resource<
8graph_attention_sparse_matmul_19_readvariableop_resource<
8graph_attention_sparse_matmul_20_readvariableop_resource<
8graph_attention_sparse_biasadd_6_readvariableop_resource<
8graph_attention_sparse_matmul_21_readvariableop_resource<
8graph_attention_sparse_matmul_22_readvariableop_resource<
8graph_attention_sparse_matmul_23_readvariableop_resource<
8graph_attention_sparse_biasadd_7_readvariableop_resource;
7graph_attention_sparse_1_matmul_readvariableop_resource=
9graph_attention_sparse_1_matmul_1_readvariableop_resource=
9graph_attention_sparse_1_matmul_2_readvariableop_resource<
8graph_attention_sparse_1_biasadd_readvariableop_resource
identity??-graph_attention_sparse/BiasAdd/ReadVariableOp?/graph_attention_sparse/BiasAdd_1/ReadVariableOp?/graph_attention_sparse/BiasAdd_2/ReadVariableOp?/graph_attention_sparse/BiasAdd_3/ReadVariableOp?/graph_attention_sparse/BiasAdd_4/ReadVariableOp?/graph_attention_sparse/BiasAdd_5/ReadVariableOp?/graph_attention_sparse/BiasAdd_6/ReadVariableOp?/graph_attention_sparse/BiasAdd_7/ReadVariableOp?,graph_attention_sparse/MatMul/ReadVariableOp?.graph_attention_sparse/MatMul_1/ReadVariableOp?/graph_attention_sparse/MatMul_10/ReadVariableOp?/graph_attention_sparse/MatMul_11/ReadVariableOp?/graph_attention_sparse/MatMul_12/ReadVariableOp?/graph_attention_sparse/MatMul_13/ReadVariableOp?/graph_attention_sparse/MatMul_14/ReadVariableOp?/graph_attention_sparse/MatMul_15/ReadVariableOp?/graph_attention_sparse/MatMul_16/ReadVariableOp?/graph_attention_sparse/MatMul_17/ReadVariableOp?/graph_attention_sparse/MatMul_18/ReadVariableOp?/graph_attention_sparse/MatMul_19/ReadVariableOp?.graph_attention_sparse/MatMul_2/ReadVariableOp?/graph_attention_sparse/MatMul_20/ReadVariableOp?/graph_attention_sparse/MatMul_21/ReadVariableOp?/graph_attention_sparse/MatMul_22/ReadVariableOp?/graph_attention_sparse/MatMul_23/ReadVariableOp?.graph_attention_sparse/MatMul_3/ReadVariableOp?.graph_attention_sparse/MatMul_4/ReadVariableOp?.graph_attention_sparse/MatMul_5/ReadVariableOp?.graph_attention_sparse/MatMul_6/ReadVariableOp?.graph_attention_sparse/MatMul_7/ReadVariableOp?.graph_attention_sparse/MatMul_8/ReadVariableOp?.graph_attention_sparse/MatMul_9/ReadVariableOp?/graph_attention_sparse_1/BiasAdd/ReadVariableOp?.graph_attention_sparse_1/MatMul/ReadVariableOp?0graph_attention_sparse_1/MatMul_1/ReadVariableOp?0graph_attention_sparse_1/MatMul_2/ReadVariableOp?
"squeezed_sparse_conversion/SqueezeSqueezeinputs_2*
T0	*'
_output_shapes
:?????????*
squeeze_dims
 2$
"squeezed_sparse_conversion/Squeeze?
$squeezed_sparse_conversion/Squeeze_1Squeezeinputs_3*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2&
$squeezed_sparse_conversion/Squeeze_1?
3squeezed_sparse_conversion/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      25
3squeezed_sparse_conversion/SparseTensor/dense_shapei
dropout/IdentityIdentityinputs_0*
T0*$
_output_shapes
:??2
dropout/Identity?
graph_attention_sparse/SqueezeSqueezedropout/Identity:output:0*
T0* 
_output_shapes
:
??*
squeeze_dims
 2 
graph_attention_sparse/Squeeze?
,graph_attention_sparse/MatMul/ReadVariableOpReadVariableOp5graph_attention_sparse_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,graph_attention_sparse/MatMul/ReadVariableOp?
graph_attention_sparse/MatMulMatMul'graph_attention_sparse/Squeeze:output:04graph_attention_sparse/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
graph_attention_sparse/MatMul?
.graph_attention_sparse/MatMul_1/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_1/ReadVariableOp?
graph_attention_sparse/MatMul_1MatMul'graph_attention_sparse/MatMul:product:06graph_attention_sparse/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_1?
.graph_attention_sparse/MatMul_2/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_2/ReadVariableOp?
graph_attention_sparse/MatMul_2MatMul'graph_attention_sparse/MatMul:product:06graph_attention_sparse/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_2?
$graph_attention_sparse/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$graph_attention_sparse/Reshape/shape?
graph_attention_sparse/ReshapeReshape)graph_attention_sparse/MatMul_1:product:0-graph_attention_sparse/Reshape/shape:output:0*
T0*
_output_shapes	
:?2 
graph_attention_sparse/Reshape?
*graph_attention_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*graph_attention_sparse/strided_slice/stack?
,graph_attention_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice/stack_1?
,graph_attention_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,graph_attention_sparse/strided_slice/stack_2?
$graph_attention_sparse/strided_sliceStridedSlice+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/strided_slice/stack:output:05graph_attention_sparse/strided_slice/stack_1:output:05graph_attention_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2&
$graph_attention_sparse/strided_slice?
$graph_attention_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$graph_attention_sparse/GatherV2/axis?
graph_attention_sparse/GatherV2GatherV2'graph_attention_sparse/Reshape:output:0-graph_attention_sparse/strided_slice:output:0-graph_attention_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2!
graph_attention_sparse/GatherV2?
&graph_attention_sparse/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_1/shape?
 graph_attention_sparse/Reshape_1Reshape)graph_attention_sparse/MatMul_2:product:0/graph_attention_sparse/Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_1?
,graph_attention_sparse/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_1/stack?
.graph_attention_sparse/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_1/stack_1?
.graph_attention_sparse/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_1/stack_2?
&graph_attention_sparse/strided_slice_1StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_1/stack:output:07graph_attention_sparse/strided_slice_1/stack_1:output:07graph_attention_sparse/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_1?
&graph_attention_sparse/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_1/axis?
!graph_attention_sparse/GatherV2_1GatherV2)graph_attention_sparse/Reshape_1:output:0/graph_attention_sparse/strided_slice_1:output:0/graph_attention_sparse/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_1?
graph_attention_sparse/addAddV2(graph_attention_sparse/GatherV2:output:0*graph_attention_sparse/GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add?
,graph_attention_sparse/leaky_re_lu/LeakyRelu	LeakyRelugraph_attention_sparse/add:z:0*#
_output_shapes
:?????????2.
,graph_attention_sparse/leaky_re_lu/LeakyRelu?
'graph_attention_sparse/dropout/IdentityIdentity'graph_attention_sparse/MatMul:product:0*
T0*
_output_shapes
:	?2)
'graph_attention_sparse/dropout/Identity?
)graph_attention_sparse/dropout_1/IdentityIdentity:graph_attention_sparse/leaky_re_lu/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2+
)graph_attention_sparse/dropout_1/Identity?
/graph_attention_sparse/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      21
/graph_attention_sparse/SparseTensor/dense_shape?
2graph_attention_sparse/SparseSoftmax/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_1/Identity:output:08graph_attention_sparse/SparseTensor/dense_shape:output:0*
T0*#
_output_shapes
:?????????24
2graph_attention_sparse/SparseSoftmax/SparseSoftmax?
Fgraph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0;graph_attention_sparse/SparseSoftmax/SparseSoftmax:output:08graph_attention_sparse/SparseTensor/dense_shape:output:00graph_attention_sparse/dropout/Identity:output:0*
T0*
_output_shapes
:	?2H
Fgraph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
-graph_attention_sparse/BiasAdd/ReadVariableOpReadVariableOp6graph_attention_sparse_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-graph_attention_sparse/BiasAdd/ReadVariableOp?
graph_attention_sparse/BiasAddBiasAddPgraph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:05graph_attention_sparse/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2 
graph_attention_sparse/BiasAdd?
.graph_attention_sparse/MatMul_3/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_3_readvariableop_resource*
_output_shapes
:	?*
dtype020
.graph_attention_sparse/MatMul_3/ReadVariableOp?
graph_attention_sparse/MatMul_3MatMul'graph_attention_sparse/Squeeze:output:06graph_attention_sparse/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_3?
.graph_attention_sparse/MatMul_4/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_4_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_4/ReadVariableOp?
graph_attention_sparse/MatMul_4MatMul)graph_attention_sparse/MatMul_3:product:06graph_attention_sparse/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_4?
.graph_attention_sparse/MatMul_5/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_5_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_5/ReadVariableOp?
graph_attention_sparse/MatMul_5MatMul)graph_attention_sparse/MatMul_3:product:06graph_attention_sparse/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_5?
&graph_attention_sparse/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_2/shape?
 graph_attention_sparse/Reshape_2Reshape)graph_attention_sparse/MatMul_4:product:0/graph_attention_sparse/Reshape_2/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_2?
,graph_attention_sparse/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_2/stack?
.graph_attention_sparse/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_2/stack_1?
.graph_attention_sparse/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_2/stack_2?
&graph_attention_sparse/strided_slice_2StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_2/stack:output:07graph_attention_sparse/strided_slice_2/stack_1:output:07graph_attention_sparse/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_2?
&graph_attention_sparse/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_2/axis?
!graph_attention_sparse/GatherV2_2GatherV2)graph_attention_sparse/Reshape_2:output:0/graph_attention_sparse/strided_slice_2:output:0/graph_attention_sparse/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_2?
&graph_attention_sparse/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_3/shape?
 graph_attention_sparse/Reshape_3Reshape)graph_attention_sparse/MatMul_5:product:0/graph_attention_sparse/Reshape_3/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_3?
,graph_attention_sparse/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_3/stack?
.graph_attention_sparse/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_3/stack_1?
.graph_attention_sparse/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_3/stack_2?
&graph_attention_sparse/strided_slice_3StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_3/stack:output:07graph_attention_sparse/strided_slice_3/stack_1:output:07graph_attention_sparse/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_3?
&graph_attention_sparse/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_3/axis?
!graph_attention_sparse/GatherV2_3GatherV2)graph_attention_sparse/Reshape_3:output:0/graph_attention_sparse/strided_slice_3:output:0/graph_attention_sparse/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_3?
graph_attention_sparse/add_1AddV2*graph_attention_sparse/GatherV2_2:output:0*graph_attention_sparse/GatherV2_3:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_1?
.graph_attention_sparse/leaky_re_lu_1/LeakyRelu	LeakyRelu graph_attention_sparse/add_1:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_1/LeakyRelu?
)graph_attention_sparse/dropout_2/IdentityIdentity)graph_attention_sparse/MatMul_3:product:0*
T0*
_output_shapes
:	?2+
)graph_attention_sparse/dropout_2/Identity?
)graph_attention_sparse/dropout_3/IdentityIdentity<graph_attention_sparse/leaky_re_lu_1/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2+
)graph_attention_sparse/dropout_3/Identity?
1graph_attention_sparse/SparseTensor_1/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_1/dense_shape?
4graph_attention_sparse/SparseSoftmax_1/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_3/Identity:output:0:graph_attention_sparse/SparseTensor_1/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_1/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_1/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_1/dense_shape:output:02graph_attention_sparse/dropout_2/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_1/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_1_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_1/ReadVariableOp?
 graph_attention_sparse/BiasAdd_1BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_1?
.graph_attention_sparse/MatMul_6/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_6_readvariableop_resource*
_output_shapes
:	?*
dtype020
.graph_attention_sparse/MatMul_6/ReadVariableOp?
graph_attention_sparse/MatMul_6MatMul'graph_attention_sparse/Squeeze:output:06graph_attention_sparse/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_6?
.graph_attention_sparse/MatMul_7/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_7_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_7/ReadVariableOp?
graph_attention_sparse/MatMul_7MatMul)graph_attention_sparse/MatMul_6:product:06graph_attention_sparse/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_7?
.graph_attention_sparse/MatMul_8/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_8_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_8/ReadVariableOp?
graph_attention_sparse/MatMul_8MatMul)graph_attention_sparse/MatMul_6:product:06graph_attention_sparse/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_8?
&graph_attention_sparse/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_4/shape?
 graph_attention_sparse/Reshape_4Reshape)graph_attention_sparse/MatMul_7:product:0/graph_attention_sparse/Reshape_4/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_4?
,graph_attention_sparse/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_4/stack?
.graph_attention_sparse/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_4/stack_1?
.graph_attention_sparse/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_4/stack_2?
&graph_attention_sparse/strided_slice_4StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_4/stack:output:07graph_attention_sparse/strided_slice_4/stack_1:output:07graph_attention_sparse/strided_slice_4/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_4?
&graph_attention_sparse/GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_4/axis?
!graph_attention_sparse/GatherV2_4GatherV2)graph_attention_sparse/Reshape_4:output:0/graph_attention_sparse/strided_slice_4:output:0/graph_attention_sparse/GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_4?
&graph_attention_sparse/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_5/shape?
 graph_attention_sparse/Reshape_5Reshape)graph_attention_sparse/MatMul_8:product:0/graph_attention_sparse/Reshape_5/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_5?
,graph_attention_sparse/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_5/stack?
.graph_attention_sparse/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_5/stack_1?
.graph_attention_sparse/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_5/stack_2?
&graph_attention_sparse/strided_slice_5StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_5/stack:output:07graph_attention_sparse/strided_slice_5/stack_1:output:07graph_attention_sparse/strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_5?
&graph_attention_sparse/GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_5/axis?
!graph_attention_sparse/GatherV2_5GatherV2)graph_attention_sparse/Reshape_5:output:0/graph_attention_sparse/strided_slice_5:output:0/graph_attention_sparse/GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_5?
graph_attention_sparse/add_2AddV2*graph_attention_sparse/GatherV2_4:output:0*graph_attention_sparse/GatherV2_5:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_2?
.graph_attention_sparse/leaky_re_lu_2/LeakyRelu	LeakyRelu graph_attention_sparse/add_2:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_2/LeakyRelu?
)graph_attention_sparse/dropout_4/IdentityIdentity)graph_attention_sparse/MatMul_6:product:0*
T0*
_output_shapes
:	?2+
)graph_attention_sparse/dropout_4/Identity?
)graph_attention_sparse/dropout_5/IdentityIdentity<graph_attention_sparse/leaky_re_lu_2/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2+
)graph_attention_sparse/dropout_5/Identity?
1graph_attention_sparse/SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_2/dense_shape?
4graph_attention_sparse/SparseSoftmax_2/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_5/Identity:output:0:graph_attention_sparse/SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_2/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_2/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_2/dense_shape:output:02graph_attention_sparse/dropout_4/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_2/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_2_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_2/ReadVariableOp?
 graph_attention_sparse/BiasAdd_2BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_2?
.graph_attention_sparse/MatMul_9/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_9_readvariableop_resource*
_output_shapes
:	?*
dtype020
.graph_attention_sparse/MatMul_9/ReadVariableOp?
graph_attention_sparse/MatMul_9MatMul'graph_attention_sparse/Squeeze:output:06graph_attention_sparse/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_9?
/graph_attention_sparse/MatMul_10/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_10_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_10/ReadVariableOp?
 graph_attention_sparse/MatMul_10MatMul)graph_attention_sparse/MatMul_9:product:07graph_attention_sparse/MatMul_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_10?
/graph_attention_sparse/MatMul_11/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_11_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_11/ReadVariableOp?
 graph_attention_sparse/MatMul_11MatMul)graph_attention_sparse/MatMul_9:product:07graph_attention_sparse/MatMul_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_11?
&graph_attention_sparse/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_6/shape?
 graph_attention_sparse/Reshape_6Reshape*graph_attention_sparse/MatMul_10:product:0/graph_attention_sparse/Reshape_6/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_6?
,graph_attention_sparse/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_6/stack?
.graph_attention_sparse/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_6/stack_1?
.graph_attention_sparse/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_6/stack_2?
&graph_attention_sparse/strided_slice_6StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_6/stack:output:07graph_attention_sparse/strided_slice_6/stack_1:output:07graph_attention_sparse/strided_slice_6/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_6?
&graph_attention_sparse/GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_6/axis?
!graph_attention_sparse/GatherV2_6GatherV2)graph_attention_sparse/Reshape_6:output:0/graph_attention_sparse/strided_slice_6:output:0/graph_attention_sparse/GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_6?
&graph_attention_sparse/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_7/shape?
 graph_attention_sparse/Reshape_7Reshape*graph_attention_sparse/MatMul_11:product:0/graph_attention_sparse/Reshape_7/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_7?
,graph_attention_sparse/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_7/stack?
.graph_attention_sparse/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_7/stack_1?
.graph_attention_sparse/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_7/stack_2?
&graph_attention_sparse/strided_slice_7StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_7/stack:output:07graph_attention_sparse/strided_slice_7/stack_1:output:07graph_attention_sparse/strided_slice_7/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_7?
&graph_attention_sparse/GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_7/axis?
!graph_attention_sparse/GatherV2_7GatherV2)graph_attention_sparse/Reshape_7:output:0/graph_attention_sparse/strided_slice_7:output:0/graph_attention_sparse/GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_7?
graph_attention_sparse/add_3AddV2*graph_attention_sparse/GatherV2_6:output:0*graph_attention_sparse/GatherV2_7:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_3?
.graph_attention_sparse/leaky_re_lu_3/LeakyRelu	LeakyRelu graph_attention_sparse/add_3:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_3/LeakyRelu?
)graph_attention_sparse/dropout_6/IdentityIdentity)graph_attention_sparse/MatMul_9:product:0*
T0*
_output_shapes
:	?2+
)graph_attention_sparse/dropout_6/Identity?
)graph_attention_sparse/dropout_7/IdentityIdentity<graph_attention_sparse/leaky_re_lu_3/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2+
)graph_attention_sparse/dropout_7/Identity?
1graph_attention_sparse/SparseTensor_3/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_3/dense_shape?
4graph_attention_sparse/SparseSoftmax_3/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_7/Identity:output:0:graph_attention_sparse/SparseTensor_3/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_3/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_3/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_3/dense_shape:output:02graph_attention_sparse/dropout_6/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_3/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_3_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_3/ReadVariableOp?
 graph_attention_sparse/BiasAdd_3BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_3?
/graph_attention_sparse/MatMul_12/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_12_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_12/ReadVariableOp?
 graph_attention_sparse/MatMul_12MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_12?
/graph_attention_sparse/MatMul_13/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_13_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_13/ReadVariableOp?
 graph_attention_sparse/MatMul_13MatMul*graph_attention_sparse/MatMul_12:product:07graph_attention_sparse/MatMul_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_13?
/graph_attention_sparse/MatMul_14/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_14_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_14/ReadVariableOp?
 graph_attention_sparse/MatMul_14MatMul*graph_attention_sparse/MatMul_12:product:07graph_attention_sparse/MatMul_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_14?
&graph_attention_sparse/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_8/shape?
 graph_attention_sparse/Reshape_8Reshape*graph_attention_sparse/MatMul_13:product:0/graph_attention_sparse/Reshape_8/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_8?
,graph_attention_sparse/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_8/stack?
.graph_attention_sparse/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_8/stack_1?
.graph_attention_sparse/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_8/stack_2?
&graph_attention_sparse/strided_slice_8StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_8/stack:output:07graph_attention_sparse/strided_slice_8/stack_1:output:07graph_attention_sparse/strided_slice_8/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_8?
&graph_attention_sparse/GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_8/axis?
!graph_attention_sparse/GatherV2_8GatherV2)graph_attention_sparse/Reshape_8:output:0/graph_attention_sparse/strided_slice_8:output:0/graph_attention_sparse/GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_8?
&graph_attention_sparse/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_9/shape?
 graph_attention_sparse/Reshape_9Reshape*graph_attention_sparse/MatMul_14:product:0/graph_attention_sparse/Reshape_9/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_9?
,graph_attention_sparse/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_9/stack?
.graph_attention_sparse/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_9/stack_1?
.graph_attention_sparse/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_9/stack_2?
&graph_attention_sparse/strided_slice_9StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_9/stack:output:07graph_attention_sparse/strided_slice_9/stack_1:output:07graph_attention_sparse/strided_slice_9/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_9?
&graph_attention_sparse/GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_9/axis?
!graph_attention_sparse/GatherV2_9GatherV2)graph_attention_sparse/Reshape_9:output:0/graph_attention_sparse/strided_slice_9:output:0/graph_attention_sparse/GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_9?
graph_attention_sparse/add_4AddV2*graph_attention_sparse/GatherV2_8:output:0*graph_attention_sparse/GatherV2_9:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_4?
.graph_attention_sparse/leaky_re_lu_4/LeakyRelu	LeakyRelu graph_attention_sparse/add_4:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_4/LeakyRelu?
)graph_attention_sparse/dropout_8/IdentityIdentity*graph_attention_sparse/MatMul_12:product:0*
T0*
_output_shapes
:	?2+
)graph_attention_sparse/dropout_8/Identity?
)graph_attention_sparse/dropout_9/IdentityIdentity<graph_attention_sparse/leaky_re_lu_4/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2+
)graph_attention_sparse/dropout_9/Identity?
1graph_attention_sparse/SparseTensor_4/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_4/dense_shape?
4graph_attention_sparse/SparseSoftmax_4/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_9/Identity:output:0:graph_attention_sparse/SparseTensor_4/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_4/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_4/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_4/dense_shape:output:02graph_attention_sparse/dropout_8/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_4/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_4_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_4/ReadVariableOp?
 graph_attention_sparse/BiasAdd_4BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_4?
/graph_attention_sparse/MatMul_15/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_15_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_15/ReadVariableOp?
 graph_attention_sparse/MatMul_15MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_15?
/graph_attention_sparse/MatMul_16/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_16_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_16/ReadVariableOp?
 graph_attention_sparse/MatMul_16MatMul*graph_attention_sparse/MatMul_15:product:07graph_attention_sparse/MatMul_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_16?
/graph_attention_sparse/MatMul_17/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_17_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_17/ReadVariableOp?
 graph_attention_sparse/MatMul_17MatMul*graph_attention_sparse/MatMul_15:product:07graph_attention_sparse/MatMul_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_17?
'graph_attention_sparse/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_10/shape?
!graph_attention_sparse/Reshape_10Reshape*graph_attention_sparse/MatMul_16:product:00graph_attention_sparse/Reshape_10/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_10?
-graph_attention_sparse/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-graph_attention_sparse/strided_slice_10/stack?
/graph_attention_sparse/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_10/stack_1?
/graph_attention_sparse/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_10/stack_2?
'graph_attention_sparse/strided_slice_10StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_10/stack:output:08graph_attention_sparse/strided_slice_10/stack_1:output:08graph_attention_sparse/strided_slice_10/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_10?
'graph_attention_sparse/GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_10/axis?
"graph_attention_sparse/GatherV2_10GatherV2*graph_attention_sparse/Reshape_10:output:00graph_attention_sparse/strided_slice_10:output:00graph_attention_sparse/GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_10?
'graph_attention_sparse/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_11/shape?
!graph_attention_sparse/Reshape_11Reshape*graph_attention_sparse/MatMul_17:product:00graph_attention_sparse/Reshape_11/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_11?
-graph_attention_sparse/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-graph_attention_sparse/strided_slice_11/stack?
/graph_attention_sparse/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_11/stack_1?
/graph_attention_sparse/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_11/stack_2?
'graph_attention_sparse/strided_slice_11StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_11/stack:output:08graph_attention_sparse/strided_slice_11/stack_1:output:08graph_attention_sparse/strided_slice_11/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_11?
'graph_attention_sparse/GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_11/axis?
"graph_attention_sparse/GatherV2_11GatherV2*graph_attention_sparse/Reshape_11:output:00graph_attention_sparse/strided_slice_11:output:00graph_attention_sparse/GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_11?
graph_attention_sparse/add_5AddV2+graph_attention_sparse/GatherV2_10:output:0+graph_attention_sparse/GatherV2_11:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_5?
.graph_attention_sparse/leaky_re_lu_5/LeakyRelu	LeakyRelu graph_attention_sparse/add_5:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_5/LeakyRelu?
*graph_attention_sparse/dropout_10/IdentityIdentity*graph_attention_sparse/MatMul_15:product:0*
T0*
_output_shapes
:	?2,
*graph_attention_sparse/dropout_10/Identity?
*graph_attention_sparse/dropout_11/IdentityIdentity<graph_attention_sparse/leaky_re_lu_5/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2,
*graph_attention_sparse/dropout_11/Identity?
1graph_attention_sparse/SparseTensor_5/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_5/dense_shape?
4graph_attention_sparse/SparseSoftmax_5/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/dropout_11/Identity:output:0:graph_attention_sparse/SparseTensor_5/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_5/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_5/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_5/dense_shape:output:03graph_attention_sparse/dropout_10/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_5/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_5_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_5/ReadVariableOp?
 graph_attention_sparse/BiasAdd_5BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_5?
/graph_attention_sparse/MatMul_18/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_18_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_18/ReadVariableOp?
 graph_attention_sparse/MatMul_18MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_18/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_18?
/graph_attention_sparse/MatMul_19/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_19_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_19/ReadVariableOp?
 graph_attention_sparse/MatMul_19MatMul*graph_attention_sparse/MatMul_18:product:07graph_attention_sparse/MatMul_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_19?
/graph_attention_sparse/MatMul_20/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_20_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_20/ReadVariableOp?
 graph_attention_sparse/MatMul_20MatMul*graph_attention_sparse/MatMul_18:product:07graph_attention_sparse/MatMul_20/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_20?
'graph_attention_sparse/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_12/shape?
!graph_attention_sparse/Reshape_12Reshape*graph_attention_sparse/MatMul_19:product:00graph_attention_sparse/Reshape_12/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_12?
-graph_attention_sparse/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-graph_attention_sparse/strided_slice_12/stack?
/graph_attention_sparse/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_12/stack_1?
/graph_attention_sparse/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_12/stack_2?
'graph_attention_sparse/strided_slice_12StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_12/stack:output:08graph_attention_sparse/strided_slice_12/stack_1:output:08graph_attention_sparse/strided_slice_12/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_12?
'graph_attention_sparse/GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_12/axis?
"graph_attention_sparse/GatherV2_12GatherV2*graph_attention_sparse/Reshape_12:output:00graph_attention_sparse/strided_slice_12:output:00graph_attention_sparse/GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_12?
'graph_attention_sparse/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_13/shape?
!graph_attention_sparse/Reshape_13Reshape*graph_attention_sparse/MatMul_20:product:00graph_attention_sparse/Reshape_13/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_13?
-graph_attention_sparse/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-graph_attention_sparse/strided_slice_13/stack?
/graph_attention_sparse/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_13/stack_1?
/graph_attention_sparse/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_13/stack_2?
'graph_attention_sparse/strided_slice_13StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_13/stack:output:08graph_attention_sparse/strided_slice_13/stack_1:output:08graph_attention_sparse/strided_slice_13/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_13?
'graph_attention_sparse/GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_13/axis?
"graph_attention_sparse/GatherV2_13GatherV2*graph_attention_sparse/Reshape_13:output:00graph_attention_sparse/strided_slice_13:output:00graph_attention_sparse/GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_13?
graph_attention_sparse/add_6AddV2+graph_attention_sparse/GatherV2_12:output:0+graph_attention_sparse/GatherV2_13:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_6?
.graph_attention_sparse/leaky_re_lu_6/LeakyRelu	LeakyRelu graph_attention_sparse/add_6:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_6/LeakyRelu?
*graph_attention_sparse/dropout_12/IdentityIdentity*graph_attention_sparse/MatMul_18:product:0*
T0*
_output_shapes
:	?2,
*graph_attention_sparse/dropout_12/Identity?
*graph_attention_sparse/dropout_13/IdentityIdentity<graph_attention_sparse/leaky_re_lu_6/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2,
*graph_attention_sparse/dropout_13/Identity?
1graph_attention_sparse/SparseTensor_6/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_6/dense_shape?
4graph_attention_sparse/SparseSoftmax_6/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/dropout_13/Identity:output:0:graph_attention_sparse/SparseTensor_6/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_6/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_6/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_6/dense_shape:output:03graph_attention_sparse/dropout_12/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_6/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_6_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_6/ReadVariableOp?
 graph_attention_sparse/BiasAdd_6BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_6?
/graph_attention_sparse/MatMul_21/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_21_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_21/ReadVariableOp?
 graph_attention_sparse/MatMul_21MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_21/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_21?
/graph_attention_sparse/MatMul_22/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_22_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_22/ReadVariableOp?
 graph_attention_sparse/MatMul_22MatMul*graph_attention_sparse/MatMul_21:product:07graph_attention_sparse/MatMul_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_22?
/graph_attention_sparse/MatMul_23/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_23_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_23/ReadVariableOp?
 graph_attention_sparse/MatMul_23MatMul*graph_attention_sparse/MatMul_21:product:07graph_attention_sparse/MatMul_23/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_23?
'graph_attention_sparse/Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_14/shape?
!graph_attention_sparse/Reshape_14Reshape*graph_attention_sparse/MatMul_22:product:00graph_attention_sparse/Reshape_14/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_14?
-graph_attention_sparse/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-graph_attention_sparse/strided_slice_14/stack?
/graph_attention_sparse/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_14/stack_1?
/graph_attention_sparse/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_14/stack_2?
'graph_attention_sparse/strided_slice_14StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_14/stack:output:08graph_attention_sparse/strided_slice_14/stack_1:output:08graph_attention_sparse/strided_slice_14/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_14?
'graph_attention_sparse/GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_14/axis?
"graph_attention_sparse/GatherV2_14GatherV2*graph_attention_sparse/Reshape_14:output:00graph_attention_sparse/strided_slice_14:output:00graph_attention_sparse/GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_14?
'graph_attention_sparse/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_15/shape?
!graph_attention_sparse/Reshape_15Reshape*graph_attention_sparse/MatMul_23:product:00graph_attention_sparse/Reshape_15/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_15?
-graph_attention_sparse/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-graph_attention_sparse/strided_slice_15/stack?
/graph_attention_sparse/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_15/stack_1?
/graph_attention_sparse/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_15/stack_2?
'graph_attention_sparse/strided_slice_15StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_15/stack:output:08graph_attention_sparse/strided_slice_15/stack_1:output:08graph_attention_sparse/strided_slice_15/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_15?
'graph_attention_sparse/GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_15/axis?
"graph_attention_sparse/GatherV2_15GatherV2*graph_attention_sparse/Reshape_15:output:00graph_attention_sparse/strided_slice_15:output:00graph_attention_sparse/GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_15?
graph_attention_sparse/add_7AddV2+graph_attention_sparse/GatherV2_14:output:0+graph_attention_sparse/GatherV2_15:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_7?
.graph_attention_sparse/leaky_re_lu_7/LeakyRelu	LeakyRelu graph_attention_sparse/add_7:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_7/LeakyRelu?
*graph_attention_sparse/dropout_14/IdentityIdentity*graph_attention_sparse/MatMul_21:product:0*
T0*
_output_shapes
:	?2,
*graph_attention_sparse/dropout_14/Identity?
*graph_attention_sparse/dropout_15/IdentityIdentity<graph_attention_sparse/leaky_re_lu_7/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2,
*graph_attention_sparse/dropout_15/Identity?
1graph_attention_sparse/SparseTensor_7/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_7/dense_shape?
4graph_attention_sparse/SparseSoftmax_7/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/dropout_15/Identity:output:0:graph_attention_sparse/SparseTensor_7/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_7/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_7/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_7/dense_shape:output:03graph_attention_sparse/dropout_14/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_7/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_7_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_7/ReadVariableOp?
 graph_attention_sparse/BiasAdd_7BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_7?
"graph_attention_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"graph_attention_sparse/concat/axis?
graph_attention_sparse/concatConcatV2'graph_attention_sparse/BiasAdd:output:0)graph_attention_sparse/BiasAdd_1:output:0)graph_attention_sparse/BiasAdd_2:output:0)graph_attention_sparse/BiasAdd_3:output:0)graph_attention_sparse/BiasAdd_4:output:0)graph_attention_sparse/BiasAdd_5:output:0)graph_attention_sparse/BiasAdd_6:output:0)graph_attention_sparse/BiasAdd_7:output:0+graph_attention_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:	?@2
graph_attention_sparse/concat?
graph_attention_sparse/EluElu&graph_attention_sparse/concat:output:0*
T0*
_output_shapes
:	?@2
graph_attention_sparse/Elu?
%graph_attention_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%graph_attention_sparse/ExpandDims/dim?
!graph_attention_sparse/ExpandDims
ExpandDims(graph_attention_sparse/Elu:activations:0.graph_attention_sparse/ExpandDims/dim:output:0*
T0*#
_output_shapes
:?@2#
!graph_attention_sparse/ExpandDims?
dropout_1/IdentityIdentity*graph_attention_sparse/ExpandDims:output:0*
T0*#
_output_shapes
:?@2
dropout_1/Identity?
 graph_attention_sparse_1/SqueezeSqueezedropout_1/Identity:output:0*
T0*
_output_shapes
:	?@*
squeeze_dims
 2"
 graph_attention_sparse_1/Squeeze?
.graph_attention_sparse_1/MatMul/ReadVariableOpReadVariableOp7graph_attention_sparse_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.graph_attention_sparse_1/MatMul/ReadVariableOp?
graph_attention_sparse_1/MatMulMatMul)graph_attention_sparse_1/Squeeze:output:06graph_attention_sparse_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse_1/MatMul?
0graph_attention_sparse_1/MatMul_1/ReadVariableOpReadVariableOp9graph_attention_sparse_1_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype022
0graph_attention_sparse_1/MatMul_1/ReadVariableOp?
!graph_attention_sparse_1/MatMul_1MatMul)graph_attention_sparse_1/MatMul:product:08graph_attention_sparse_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!graph_attention_sparse_1/MatMul_1?
0graph_attention_sparse_1/MatMul_2/ReadVariableOpReadVariableOp9graph_attention_sparse_1_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype022
0graph_attention_sparse_1/MatMul_2/ReadVariableOp?
!graph_attention_sparse_1/MatMul_2MatMul)graph_attention_sparse_1/MatMul:product:08graph_attention_sparse_1/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!graph_attention_sparse_1/MatMul_2?
&graph_attention_sparse_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse_1/Reshape/shape?
 graph_attention_sparse_1/ReshapeReshape+graph_attention_sparse_1/MatMul_1:product:0/graph_attention_sparse_1/Reshape/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse_1/Reshape?
,graph_attention_sparse_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse_1/strided_slice/stack?
.graph_attention_sparse_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse_1/strided_slice/stack_1?
.graph_attention_sparse_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse_1/strided_slice/stack_2?
&graph_attention_sparse_1/strided_sliceStridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse_1/strided_slice/stack:output:07graph_attention_sparse_1/strided_slice/stack_1:output:07graph_attention_sparse_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse_1/strided_slice?
&graph_attention_sparse_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse_1/GatherV2/axis?
!graph_attention_sparse_1/GatherV2GatherV2)graph_attention_sparse_1/Reshape:output:0/graph_attention_sparse_1/strided_slice:output:0/graph_attention_sparse_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse_1/GatherV2?
(graph_attention_sparse_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(graph_attention_sparse_1/Reshape_1/shape?
"graph_attention_sparse_1/Reshape_1Reshape+graph_attention_sparse_1/MatMul_2:product:01graph_attention_sparse_1/Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2$
"graph_attention_sparse_1/Reshape_1?
.graph_attention_sparse_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse_1/strided_slice_1/stack?
0graph_attention_sparse_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0graph_attention_sparse_1/strided_slice_1/stack_1?
0graph_attention_sparse_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0graph_attention_sparse_1/strided_slice_1/stack_2?
(graph_attention_sparse_1/strided_slice_1StridedSlice+squeezed_sparse_conversion/Squeeze:output:07graph_attention_sparse_1/strided_slice_1/stack:output:09graph_attention_sparse_1/strided_slice_1/stack_1:output:09graph_attention_sparse_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2*
(graph_attention_sparse_1/strided_slice_1?
(graph_attention_sparse_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(graph_attention_sparse_1/GatherV2_1/axis?
#graph_attention_sparse_1/GatherV2_1GatherV2+graph_attention_sparse_1/Reshape_1:output:01graph_attention_sparse_1/strided_slice_1:output:01graph_attention_sparse_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2%
#graph_attention_sparse_1/GatherV2_1?
graph_attention_sparse_1/addAddV2*graph_attention_sparse_1/GatherV2:output:0,graph_attention_sparse_1/GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse_1/add?
0graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu	LeakyRelu graph_attention_sparse_1/add:z:0*#
_output_shapes
:?????????22
0graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu?
,graph_attention_sparse_1/dropout_16/IdentityIdentity)graph_attention_sparse_1/MatMul:product:0*
T0*
_output_shapes
:	?2.
,graph_attention_sparse_1/dropout_16/Identity?
,graph_attention_sparse_1/dropout_17/IdentityIdentity>graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2.
,graph_attention_sparse_1/dropout_17/Identity?
1graph_attention_sparse_1/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse_1/SparseTensor/dense_shape?
4graph_attention_sparse_1/SparseSoftmax/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse_1/dropout_17/Identity:output:0:graph_attention_sparse_1/SparseTensor/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse_1/SparseSoftmax/SparseSoftmax?
Hgraph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse_1/SparseSoftmax/SparseSoftmax:output:0:graph_attention_sparse_1/SparseTensor/dense_shape:output:05graph_attention_sparse_1/dropout_16/Identity:output:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
/graph_attention_sparse_1/BiasAdd/ReadVariableOpReadVariableOp8graph_attention_sparse_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse_1/BiasAdd/ReadVariableOp?
 graph_attention_sparse_1/BiasAddBiasAddRgraph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:07graph_attention_sparse_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse_1/BiasAdd?
graph_attention_sparse_1/stackPack)graph_attention_sparse_1/BiasAdd:output:0*
N*
T0*#
_output_shapes
:?2 
graph_attention_sparse_1/stack?
/graph_attention_sparse_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 21
/graph_attention_sparse_1/Mean/reduction_indices?
graph_attention_sparse_1/MeanMean'graph_attention_sparse_1/stack:output:08graph_attention_sparse_1/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	?2
graph_attention_sparse_1/Mean?
 graph_attention_sparse_1/SoftmaxSoftmax&graph_attention_sparse_1/Mean:output:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse_1/Softmax?
'graph_attention_sparse_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse_1/ExpandDims/dim?
#graph_attention_sparse_1/ExpandDims
ExpandDims*graph_attention_sparse_1/Softmax:softmax:00graph_attention_sparse_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:?2%
#graph_attention_sparse_1/ExpandDims~
gather_indices/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
gather_indices/GatherV2/axis?
gather_indices/GatherV2GatherV2,graph_attention_sparse_1/ExpandDims:output:0inputs_1%gather_indices/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????*

batch_dims2
gather_indices/GatherV2?
IdentityIdentity gather_indices/GatherV2:output:0.^graph_attention_sparse/BiasAdd/ReadVariableOp0^graph_attention_sparse/BiasAdd_1/ReadVariableOp0^graph_attention_sparse/BiasAdd_2/ReadVariableOp0^graph_attention_sparse/BiasAdd_3/ReadVariableOp0^graph_attention_sparse/BiasAdd_4/ReadVariableOp0^graph_attention_sparse/BiasAdd_5/ReadVariableOp0^graph_attention_sparse/BiasAdd_6/ReadVariableOp0^graph_attention_sparse/BiasAdd_7/ReadVariableOp-^graph_attention_sparse/MatMul/ReadVariableOp/^graph_attention_sparse/MatMul_1/ReadVariableOp0^graph_attention_sparse/MatMul_10/ReadVariableOp0^graph_attention_sparse/MatMul_11/ReadVariableOp0^graph_attention_sparse/MatMul_12/ReadVariableOp0^graph_attention_sparse/MatMul_13/ReadVariableOp0^graph_attention_sparse/MatMul_14/ReadVariableOp0^graph_attention_sparse/MatMul_15/ReadVariableOp0^graph_attention_sparse/MatMul_16/ReadVariableOp0^graph_attention_sparse/MatMul_17/ReadVariableOp0^graph_attention_sparse/MatMul_18/ReadVariableOp0^graph_attention_sparse/MatMul_19/ReadVariableOp/^graph_attention_sparse/MatMul_2/ReadVariableOp0^graph_attention_sparse/MatMul_20/ReadVariableOp0^graph_attention_sparse/MatMul_21/ReadVariableOp0^graph_attention_sparse/MatMul_22/ReadVariableOp0^graph_attention_sparse/MatMul_23/ReadVariableOp/^graph_attention_sparse/MatMul_3/ReadVariableOp/^graph_attention_sparse/MatMul_4/ReadVariableOp/^graph_attention_sparse/MatMul_5/ReadVariableOp/^graph_attention_sparse/MatMul_6/ReadVariableOp/^graph_attention_sparse/MatMul_7/ReadVariableOp/^graph_attention_sparse/MatMul_8/ReadVariableOp/^graph_attention_sparse/MatMul_9/ReadVariableOp0^graph_attention_sparse_1/BiasAdd/ReadVariableOp/^graph_attention_sparse_1/MatMul/ReadVariableOp1^graph_attention_sparse_1/MatMul_1/ReadVariableOp1^graph_attention_sparse_1/MatMul_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::2^
-graph_attention_sparse/BiasAdd/ReadVariableOp-graph_attention_sparse/BiasAdd/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_1/ReadVariableOp/graph_attention_sparse/BiasAdd_1/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_2/ReadVariableOp/graph_attention_sparse/BiasAdd_2/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_3/ReadVariableOp/graph_attention_sparse/BiasAdd_3/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_4/ReadVariableOp/graph_attention_sparse/BiasAdd_4/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_5/ReadVariableOp/graph_attention_sparse/BiasAdd_5/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_6/ReadVariableOp/graph_attention_sparse/BiasAdd_6/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_7/ReadVariableOp/graph_attention_sparse/BiasAdd_7/ReadVariableOp2\
,graph_attention_sparse/MatMul/ReadVariableOp,graph_attention_sparse/MatMul/ReadVariableOp2`
.graph_attention_sparse/MatMul_1/ReadVariableOp.graph_attention_sparse/MatMul_1/ReadVariableOp2b
/graph_attention_sparse/MatMul_10/ReadVariableOp/graph_attention_sparse/MatMul_10/ReadVariableOp2b
/graph_attention_sparse/MatMul_11/ReadVariableOp/graph_attention_sparse/MatMul_11/ReadVariableOp2b
/graph_attention_sparse/MatMul_12/ReadVariableOp/graph_attention_sparse/MatMul_12/ReadVariableOp2b
/graph_attention_sparse/MatMul_13/ReadVariableOp/graph_attention_sparse/MatMul_13/ReadVariableOp2b
/graph_attention_sparse/MatMul_14/ReadVariableOp/graph_attention_sparse/MatMul_14/ReadVariableOp2b
/graph_attention_sparse/MatMul_15/ReadVariableOp/graph_attention_sparse/MatMul_15/ReadVariableOp2b
/graph_attention_sparse/MatMul_16/ReadVariableOp/graph_attention_sparse/MatMul_16/ReadVariableOp2b
/graph_attention_sparse/MatMul_17/ReadVariableOp/graph_attention_sparse/MatMul_17/ReadVariableOp2b
/graph_attention_sparse/MatMul_18/ReadVariableOp/graph_attention_sparse/MatMul_18/ReadVariableOp2b
/graph_attention_sparse/MatMul_19/ReadVariableOp/graph_attention_sparse/MatMul_19/ReadVariableOp2`
.graph_attention_sparse/MatMul_2/ReadVariableOp.graph_attention_sparse/MatMul_2/ReadVariableOp2b
/graph_attention_sparse/MatMul_20/ReadVariableOp/graph_attention_sparse/MatMul_20/ReadVariableOp2b
/graph_attention_sparse/MatMul_21/ReadVariableOp/graph_attention_sparse/MatMul_21/ReadVariableOp2b
/graph_attention_sparse/MatMul_22/ReadVariableOp/graph_attention_sparse/MatMul_22/ReadVariableOp2b
/graph_attention_sparse/MatMul_23/ReadVariableOp/graph_attention_sparse/MatMul_23/ReadVariableOp2`
.graph_attention_sparse/MatMul_3/ReadVariableOp.graph_attention_sparse/MatMul_3/ReadVariableOp2`
.graph_attention_sparse/MatMul_4/ReadVariableOp.graph_attention_sparse/MatMul_4/ReadVariableOp2`
.graph_attention_sparse/MatMul_5/ReadVariableOp.graph_attention_sparse/MatMul_5/ReadVariableOp2`
.graph_attention_sparse/MatMul_6/ReadVariableOp.graph_attention_sparse/MatMul_6/ReadVariableOp2`
.graph_attention_sparse/MatMul_7/ReadVariableOp.graph_attention_sparse/MatMul_7/ReadVariableOp2`
.graph_attention_sparse/MatMul_8/ReadVariableOp.graph_attention_sparse/MatMul_8/ReadVariableOp2`
.graph_attention_sparse/MatMul_9/ReadVariableOp.graph_attention_sparse/MatMul_9/ReadVariableOp2b
/graph_attention_sparse_1/BiasAdd/ReadVariableOp/graph_attention_sparse_1/BiasAdd/ReadVariableOp2`
.graph_attention_sparse_1/MatMul/ReadVariableOp.graph_attention_sparse_1/MatMul/ReadVariableOp2d
0graph_attention_sparse_1/MatMul_1/ReadVariableOp0graph_attention_sparse_1/MatMul_1/ReadVariableOp2d
0graph_attention_sparse_1/MatMul_2/ReadVariableOp0graph_attention_sparse_1/MatMul_2/ReadVariableOp:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
?4
?
?__inference_model_layer_call_and_return_conditional_losses_9964

inputs
inputs_1
inputs_2	
inputs_3
graph_attention_sparse_9886
graph_attention_sparse_9888
graph_attention_sparse_9890
graph_attention_sparse_9892
graph_attention_sparse_9894
graph_attention_sparse_9896
graph_attention_sparse_9898
graph_attention_sparse_9900
graph_attention_sparse_9902
graph_attention_sparse_9904
graph_attention_sparse_9906
graph_attention_sparse_9908
graph_attention_sparse_9910
graph_attention_sparse_9912
graph_attention_sparse_9914
graph_attention_sparse_9916
graph_attention_sparse_9918
graph_attention_sparse_9920
graph_attention_sparse_9922
graph_attention_sparse_9924
graph_attention_sparse_9926
graph_attention_sparse_9928
graph_attention_sparse_9930
graph_attention_sparse_9932
graph_attention_sparse_9934
graph_attention_sparse_9936
graph_attention_sparse_9938
graph_attention_sparse_9940
graph_attention_sparse_9942
graph_attention_sparse_9944
graph_attention_sparse_9946
graph_attention_sparse_9948!
graph_attention_sparse_1_9952!
graph_attention_sparse_1_9954!
graph_attention_sparse_1_9956!
graph_attention_sparse_1_9958
identity??.graph_attention_sparse/StatefulPartitionedCall?0graph_attention_sparse_1/StatefulPartitionedCall?
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinputs_2inputs_3*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:?????????:?????????:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_84372,
*squeezed_sparse_conversion/PartitionedCall?
dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_84692
dropout/PartitionedCall?
.graph_attention_sparse/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_9886graph_attention_sparse_9888graph_attention_sparse_9890graph_attention_sparse_9892graph_attention_sparse_9894graph_attention_sparse_9896graph_attention_sparse_9898graph_attention_sparse_9900graph_attention_sparse_9902graph_attention_sparse_9904graph_attention_sparse_9906graph_attention_sparse_9908graph_attention_sparse_9910graph_attention_sparse_9912graph_attention_sparse_9914graph_attention_sparse_9916graph_attention_sparse_9918graph_attention_sparse_9920graph_attention_sparse_9922graph_attention_sparse_9924graph_attention_sparse_9926graph_attention_sparse_9928graph_attention_sparse_9930graph_attention_sparse_9932graph_attention_sparse_9934graph_attention_sparse_9936graph_attention_sparse_9938graph_attention_sparse_9940graph_attention_sparse_9942graph_attention_sparse_9944graph_attention_sparse_9946graph_attention_sparse_9948*/
Tin(
&2$		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@*B
_read_only_resource_inputs$
" 	
 !"#*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_918020
.graph_attention_sparse/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall7graph_attention_sparse/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_94082
dropout_1/PartitionedCall?
0graph_attention_sparse_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_attention_sparse_1_9952graph_attention_sparse_1_9954graph_attention_sparse_1_9956graph_attention_sparse_1_9958*
Tin

2		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_953322
0graph_attention_sparse_1/StatefulPartitionedCall?
gather_indices/PartitionedCallPartitionedCall9graph_attention_sparse_1/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gather_indices_layer_call_and_return_conditional_losses_95832 
gather_indices/PartitionedCall?
lambda/PartitionedCallPartitionedCall'gather_indices/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_96002
lambda/PartitionedCall?
IdentityIdentitylambda/PartitionedCall:output:0/^graph_attention_sparse/StatefulPartitionedCall1^graph_attention_sparse_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::2`
.graph_attention_sparse/StatefulPartitionedCall.graph_attention_sparse/StatefulPartitionedCall2d
0graph_attention_sparse_1/StatefulPartitionedCall0graph_attention_sparse_1/StatefulPartitionedCall:L H
$
_output_shapes
:??
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_10129
input_1
input_2
input_3	
input_4
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_84202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
$
_output_shapes
:??
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?E
?
S__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_12092
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOph
SqueezeSqueezeinputs_0*
T0*
_output_shapes
:	?@*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulMatMul:product:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul#leaky_re_lu/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputsdropout_1/dropout/Mul_1:z:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd_
stackPackBiasAdd:output:0*
N*
T0*#
_output_shapes
:?2
stackr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indiceso
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:	?2
MeanV
SoftmaxSoftmaxMean:output:0*
T0*
_output_shapes
:	?2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp*
T0*#
_output_shapes
:?2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?@:?????????:?????????:::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp:M I
#
_output_shapes
:?@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_11135

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_84642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:??2

Identity"
identityIdentity:output:0*#
_input_shapes
:??22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:??
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_12019

inputs

identity_1V
IdentityIdentityinputs*
T0*#
_output_shapes
:?@2

Identitye

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
:?@2

Identity_1"!

identity_1Identity_1:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_12014

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consto
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
:?@2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?
  @   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
:?@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?@2
dropout/GreaterEqual{
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?@2
dropout/Castv
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*#
_output_shapes
:?@2
dropout/Mul_1a
IdentityIdentitydropout/Mul_1:z:0*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?3
?
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_9533

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOpf
SqueezeSqueezeinputs*
T0*
_output_shapes
:	?@*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelul
dropout/IdentityIdentityMatMul:product:0*
T0*
_output_shapes
:	?2
dropout/Identity?
dropout_1/IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????2
dropout_1/Identity?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputs_1dropout_1/Identity:output:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/Identity:output:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd_
stackPackBiasAdd:output:0*
N*
T0*#
_output_shapes
:?2
stackr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indiceso
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:	?2
MeanV
SoftmaxSoftmaxMean:output:0*
T0*
_output_shapes
:	?2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp*
T0*#
_output_shapes
:?2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?@:?????????:?????????:::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?	
?
T__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_8437

inputs	
inputs_1
identity	

identity_1

identity_2	n
SqueezeSqueezeinputs*
T0	*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezep
	Squeeze_1Squeezeinputs_1*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1?
SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor/dense_shaped
IdentityIdentitySqueeze:output:0*
T0	*'
_output_shapes
:?????????2

Identityf

Identity_1IdentitySqueeze_1:output:0*
T0*#
_output_shapes
:?????????2

Identity_1l

Identity_2Identity!SparseTensor/dense_shape:output:0*
T0	*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*=
_input_shapes,
*:?????????:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_10039
input_1
input_2
input_3	
input_4
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_99642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
$
_output_shapes
:??
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_8469

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:??2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:??2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:??:L H
$
_output_shapes
:??
 
_user_specified_nameinputs
?
\
@__inference_lambda_layer_call_and_return_conditional_losses_9600

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
I__inference_gather_indices_layer_call_and_return_conditional_losses_12180
inputs_0
inputs_1
identity`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2inputs_0inputs_1GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????*

batch_dims2

GatherV2i
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*5
_input_shapes$
":?:?????????:M I
#
_output_shapes
:?
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
8__inference_graph_attention_sparse_1_layer_call_fn_12173
inputs_0

inputs	
inputs_1
inputs_2	
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2*
Tin

2		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_95332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?@:?????????:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
]
A__inference_lambda_layer_call_and_return_conditional_losses_12194

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_lambda_layer_call_fn_12204

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_96002
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
Q__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_11565
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource%
!biasadd_1_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource$
 matmul_8_readvariableop_resource%
!biasadd_2_readvariableop_resource$
 matmul_9_readvariableop_resource%
!matmul_10_readvariableop_resource%
!matmul_11_readvariableop_resource%
!biasadd_3_readvariableop_resource%
!matmul_12_readvariableop_resource%
!matmul_13_readvariableop_resource%
!matmul_14_readvariableop_resource%
!biasadd_4_readvariableop_resource%
!matmul_15_readvariableop_resource%
!matmul_16_readvariableop_resource%
!matmul_17_readvariableop_resource%
!biasadd_5_readvariableop_resource%
!matmul_18_readvariableop_resource%
!matmul_19_readvariableop_resource%
!matmul_20_readvariableop_resource%
!biasadd_6_readvariableop_resource%
!matmul_21_readvariableop_resource%
!matmul_22_readvariableop_resource%
!matmul_23_readvariableop_resource%
!biasadd_7_readvariableop_resource
identity??BiasAdd/ReadVariableOp?BiasAdd_1/ReadVariableOp?BiasAdd_2/ReadVariableOp?BiasAdd_3/ReadVariableOp?BiasAdd_4/ReadVariableOp?BiasAdd_5/ReadVariableOp?BiasAdd_6/ReadVariableOp?BiasAdd_7/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_10/ReadVariableOp?MatMul_11/ReadVariableOp?MatMul_12/ReadVariableOp?MatMul_13/ReadVariableOp?MatMul_14/ReadVariableOp?MatMul_15/ReadVariableOp?MatMul_16/ReadVariableOp?MatMul_17/ReadVariableOp?MatMul_18/ReadVariableOp?MatMul_19/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_20/ReadVariableOp?MatMul_21/ReadVariableOp?MatMul_22/ReadVariableOp?MatMul_23/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?MatMul_8/ReadVariableOp?MatMul_9/ReadVariableOpi
SqueezeSqueezeinputs_0*
T0* 
_output_shapes
:
??*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulMatMul:product:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul#leaky_re_lu/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputsdropout_1/dropout/Mul_1:z:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_3/ReadVariableOp{
MatMul_3MatMulSqueeze:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_3?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_4/ReadVariableOp}
MatMul_4MatMulMatMul_3:product:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp}
MatMul_5MatMulMatMul_3:product:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_5u
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_2/shapeu
	Reshape_2ReshapeMatMul_4:product:0Reshape_2/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_2:output:0strided_slice_2:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_2u
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_3/shapeu
	Reshape_3ReshapeMatMul_5:product:0Reshape_3/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_3:output:0strided_slice_3:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_3o
add_1AddV2GatherV2_2:output:0GatherV2_3:output:0*
T0*#
_output_shapes
:?????????2
add_1o
leaky_re_lu_1/LeakyRelu	LeakyRelu	add_1:z:0*#
_output_shapes
:?????????2
leaky_re_lu_1/LeakyReluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulMatMul_3:product:0 dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_2/dropout/Mul_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const?
dropout_3/dropout/MulMul%leaky_re_lu_1/LeakyRelu:activations:0 dropout_3/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_3/dropout/Mul_1?
SparseTensor_3/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_3/dense_shape?
SparseSoftmax_1/SparseSoftmaxSparseSoftmaxinputsdropout_3/dropout/Mul_1:z:0#SparseTensor_3/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_1/SparseSoftmax?
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_1/SparseSoftmax:output:0#SparseTensor_3/dense_shape:output:0dropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMul?
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_1/ReadVariableOp?
	BiasAdd_1BiasAdd;SparseTensorDenseMatMul_1/SparseTensorDenseMatMul:product:0 BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_1?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_6/ReadVariableOp{
MatMul_6MatMulSqueeze:output:0MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_7/ReadVariableOp}
MatMul_7MatMulMatMul_6:product:0MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_7?
MatMul_8/ReadVariableOpReadVariableOp matmul_8_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_8/ReadVariableOp}
MatMul_8MatMulMatMul_6:product:0MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_8u
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_4/shapeu
	Reshape_4ReshapeMatMul_7:product:0Reshape_4/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_4
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_4:output:0strided_slice_4:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_4u
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_5/shapeu
	Reshape_5ReshapeMatMul_8:product:0Reshape_5/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_5
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_5:output:0strided_slice_5:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_5o
add_2AddV2GatherV2_4:output:0GatherV2_5:output:0*
T0*#
_output_shapes
:?????????2
add_2o
leaky_re_lu_2/LeakyRelu	LeakyRelu	add_2:z:0*#
_output_shapes
:?????????2
leaky_re_lu_2/LeakyReluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulMatMul_6:product:0 dropout_4/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_4/dropout/Mul_1w
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul%leaky_re_lu_2/LeakyRelu:activations:0 dropout_5/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_5/dropout/Mul?
dropout_5/dropout/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_5/dropout/Mul_1?
SparseTensor_4/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_4/dense_shape?
SparseSoftmax_2/SparseSoftmaxSparseSoftmaxinputsdropout_5/dropout/Mul_1:z:0#SparseTensor_4/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_2/SparseSoftmax?
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_2/SparseSoftmax:output:0#SparseTensor_4/dense_shape:output:0dropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMul?
BiasAdd_2/ReadVariableOpReadVariableOp!biasadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_2/ReadVariableOp?
	BiasAdd_2BiasAdd;SparseTensorDenseMatMul_2/SparseTensorDenseMatMul:product:0 BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_2?
MatMul_9/ReadVariableOpReadVariableOp matmul_9_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_9/ReadVariableOp{
MatMul_9MatMulSqueeze:output:0MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_9?
MatMul_10/ReadVariableOpReadVariableOp!matmul_10_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_10/ReadVariableOp?
	MatMul_10MatMulMatMul_9:product:0 MatMul_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_10?
MatMul_11/ReadVariableOpReadVariableOp!matmul_11_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_11/ReadVariableOp?
	MatMul_11MatMulMatMul_9:product:0 MatMul_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_11u
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_6/shapev
	Reshape_6ReshapeMatMul_10:product:0Reshape_6/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_6:output:0strided_slice_6:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_6u
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_7/shapev
	Reshape_7ReshapeMatMul_11:product:0Reshape_7/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceinputsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_7:output:0strided_slice_7:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_7o
add_3AddV2GatherV2_6:output:0GatherV2_7:output:0*
T0*#
_output_shapes
:?????????2
add_3o
leaky_re_lu_3/LeakyRelu	LeakyRelu	add_3:z:0*#
_output_shapes
:?????????2
leaky_re_lu_3/LeakyReluw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_6/dropout/Const?
dropout_6/dropout/MulMulMatMul_9:product:0 dropout_6/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_6/dropout/Mul?
dropout_6/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_6/dropout/Mul_1w
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_7/dropout/Const?
dropout_7/dropout/MulMul%leaky_re_lu_3/LeakyRelu:activations:0 dropout_7/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_7/dropout/Mul?
dropout_7/dropout/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_7/dropout/Mul_1?
SparseTensor_5/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_5/dense_shape?
SparseSoftmax_3/SparseSoftmaxSparseSoftmaxinputsdropout_7/dropout/Mul_1:z:0#SparseTensor_5/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_3/SparseSoftmax?
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_3/SparseSoftmax:output:0#SparseTensor_5/dense_shape:output:0dropout_6/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMul?
BiasAdd_3/ReadVariableOpReadVariableOp!biasadd_3_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_3/ReadVariableOp?
	BiasAdd_3BiasAdd;SparseTensorDenseMatMul_3/SparseTensorDenseMatMul:product:0 BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_3?
MatMul_12/ReadVariableOpReadVariableOp!matmul_12_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_12/ReadVariableOp~
	MatMul_12MatMulSqueeze:output:0 MatMul_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_12?
MatMul_13/ReadVariableOpReadVariableOp!matmul_13_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_13/ReadVariableOp?
	MatMul_13MatMulMatMul_12:product:0 MatMul_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_13?
MatMul_14/ReadVariableOpReadVariableOp!matmul_14_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_14/ReadVariableOp?
	MatMul_14MatMulMatMul_12:product:0 MatMul_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_14u
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_8/shapev
	Reshape_8ReshapeMatMul_13:product:0Reshape_8/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceinputsstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8d
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axis?

GatherV2_8GatherV2Reshape_8:output:0strided_slice_8:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_8u
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_9/shapev
	Reshape_9ReshapeMatMul_14:product:0Reshape_9/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_9
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceinputsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9d
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axis?

GatherV2_9GatherV2Reshape_9:output:0strided_slice_9:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_9o
add_4AddV2GatherV2_8:output:0GatherV2_9:output:0*
T0*#
_output_shapes
:?????????2
add_4o
leaky_re_lu_4/LeakyRelu	LeakyRelu	add_4:z:0*#
_output_shapes
:?????????2
leaky_re_lu_4/LeakyReluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_8/dropout/Const?
dropout_8/dropout/MulMulMatMul_12:product:0 dropout_8/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_8/dropout/Mul_1w
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul%leaky_re_lu_4/LeakyRelu:activations:0 dropout_9/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_9/dropout/Mul_1?
SparseTensor_6/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_6/dense_shape?
SparseSoftmax_4/SparseSoftmaxSparseSoftmaxinputsdropout_9/dropout/Mul_1:z:0#SparseTensor_6/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_4/SparseSoftmax?
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_4/SparseSoftmax:output:0#SparseTensor_6/dense_shape:output:0dropout_8/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMul?
BiasAdd_4/ReadVariableOpReadVariableOp!biasadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_4/ReadVariableOp?
	BiasAdd_4BiasAdd;SparseTensorDenseMatMul_4/SparseTensorDenseMatMul:product:0 BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_4?
MatMul_15/ReadVariableOpReadVariableOp!matmul_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_15/ReadVariableOp~
	MatMul_15MatMulSqueeze:output:0 MatMul_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_15?
MatMul_16/ReadVariableOpReadVariableOp!matmul_16_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_16/ReadVariableOp?
	MatMul_16MatMulMatMul_15:product:0 MatMul_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_16?
MatMul_17/ReadVariableOpReadVariableOp!matmul_17_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_17/ReadVariableOp?
	MatMul_17MatMulMatMul_15:product:0 MatMul_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_17w
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_10/shapey

Reshape_10ReshapeMatMul_16:product:0Reshape_10/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_10?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceinputsstrided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10f
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axis?
GatherV2_10GatherV2Reshape_10:output:0strided_slice_10:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_10w
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_11/shapey

Reshape_11ReshapeMatMul_17:product:0Reshape_11/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_11?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceinputsstrided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11f
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axis?
GatherV2_11GatherV2Reshape_11:output:0strided_slice_11:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_11q
add_5AddV2GatherV2_10:output:0GatherV2_11:output:0*
T0*#
_output_shapes
:?????????2
add_5o
leaky_re_lu_5/LeakyRelu	LeakyRelu	add_5:z:0*#
_output_shapes
:?????????2
leaky_re_lu_5/LeakyReluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Const?
dropout_10/dropout/MulMulMatMul_15:product:0!dropout_10/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_10/dropout/Mul?
dropout_10/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform?
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/y?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_10/dropout/Mul_1y
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Const?
dropout_11/dropout/MulMul%leaky_re_lu_5/LeakyRelu:activations:0!dropout_11/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_11/dropout/Mul?
dropout_11/dropout/ShapeShape%leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform?
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/y?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2!
dropout_11/dropout/GreaterEqual?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_11/dropout/Cast?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_11/dropout/Mul_1?
SparseTensor_7/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_7/dense_shape?
SparseSoftmax_5/SparseSoftmaxSparseSoftmaxinputsdropout_11/dropout/Mul_1:z:0#SparseTensor_7/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_5/SparseSoftmax?
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_5/SparseSoftmax:output:0#SparseTensor_7/dense_shape:output:0dropout_10/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMul?
BiasAdd_5/ReadVariableOpReadVariableOp!biasadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_5/ReadVariableOp?
	BiasAdd_5BiasAdd;SparseTensorDenseMatMul_5/SparseTensorDenseMatMul:product:0 BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_5?
MatMul_18/ReadVariableOpReadVariableOp!matmul_18_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_18/ReadVariableOp~
	MatMul_18MatMulSqueeze:output:0 MatMul_18/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_18?
MatMul_19/ReadVariableOpReadVariableOp!matmul_19_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_19/ReadVariableOp?
	MatMul_19MatMulMatMul_18:product:0 MatMul_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_19?
MatMul_20/ReadVariableOpReadVariableOp!matmul_20_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_20/ReadVariableOp?
	MatMul_20MatMulMatMul_18:product:0 MatMul_20/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_20w
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_12/shapey

Reshape_12ReshapeMatMul_19:product:0Reshape_12/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_12?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSliceinputsstrided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12f
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axis?
GatherV2_12GatherV2Reshape_12:output:0strided_slice_12:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_12w
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_13/shapey

Reshape_13ReshapeMatMul_20:product:0Reshape_13/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_13?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSliceinputsstrided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13f
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axis?
GatherV2_13GatherV2Reshape_13:output:0strided_slice_13:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_13q
add_6AddV2GatherV2_12:output:0GatherV2_13:output:0*
T0*#
_output_shapes
:?????????2
add_6o
leaky_re_lu_6/LeakyRelu	LeakyRelu	add_6:z:0*#
_output_shapes
:?????????2
leaky_re_lu_6/LeakyReluy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_12/dropout/Const?
dropout_12/dropout/MulMulMatMul_18:product:0!dropout_12/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_12/dropout/Mul?
dropout_12/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_12/dropout/Shape?
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform?
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_12/dropout/GreaterEqual/y?
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2!
dropout_12/dropout/GreaterEqual?
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_12/dropout/Cast?
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_12/dropout/Mul_1y
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Const?
dropout_13/dropout/MulMul%leaky_re_lu_6/LeakyRelu:activations:0!dropout_13/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_13/dropout/Mul?
dropout_13/dropout/ShapeShape%leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform?
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/y?
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2!
dropout_13/dropout/GreaterEqual?
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_13/dropout/Cast?
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_13/dropout/Mul_1?
SparseTensor_8/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_8/dense_shape?
SparseSoftmax_6/SparseSoftmaxSparseSoftmaxinputsdropout_13/dropout/Mul_1:z:0#SparseTensor_8/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_6/SparseSoftmax?
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_6/SparseSoftmax:output:0#SparseTensor_8/dense_shape:output:0dropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMul?
BiasAdd_6/ReadVariableOpReadVariableOp!biasadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_6/ReadVariableOp?
	BiasAdd_6BiasAdd;SparseTensorDenseMatMul_6/SparseTensorDenseMatMul:product:0 BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_6?
MatMul_21/ReadVariableOpReadVariableOp!matmul_21_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_21/ReadVariableOp~
	MatMul_21MatMulSqueeze:output:0 MatMul_21/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_21?
MatMul_22/ReadVariableOpReadVariableOp!matmul_22_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_22/ReadVariableOp?
	MatMul_22MatMulMatMul_21:product:0 MatMul_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_22?
MatMul_23/ReadVariableOpReadVariableOp!matmul_23_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_23/ReadVariableOp?
	MatMul_23MatMulMatMul_21:product:0 MatMul_23/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_23w
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_14/shapey

Reshape_14ReshapeMatMul_22:product:0Reshape_14/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_14?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSliceinputsstrided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14f
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axis?
GatherV2_14GatherV2Reshape_14:output:0strided_slice_14:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_14w
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_15/shapey

Reshape_15ReshapeMatMul_23:product:0Reshape_15/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_15?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSliceinputsstrided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15f
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axis?
GatherV2_15GatherV2Reshape_15:output:0strided_slice_15:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_15q
add_7AddV2GatherV2_14:output:0GatherV2_15:output:0*
T0*#
_output_shapes
:?????????2
add_7o
leaky_re_lu_7/LeakyRelu	LeakyRelu	add_7:z:0*#
_output_shapes
:?????????2
leaky_re_lu_7/LeakyReluy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_14/dropout/Const?
dropout_14/dropout/MulMulMatMul_21:product:0!dropout_14/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_14/dropout/Mul?
dropout_14/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_14/dropout/Shape?
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype021
/dropout_14/dropout/random_uniform/RandomUniform?
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_14/dropout/GreaterEqual/y?
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2!
dropout_14/dropout/GreaterEqual?
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_14/dropout/Cast?
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_14/dropout/Mul_1y
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_15/dropout/Const?
dropout_15/dropout/MulMul%leaky_re_lu_7/LeakyRelu:activations:0!dropout_15/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_15/dropout/Mul?
dropout_15/dropout/ShapeShape%leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape?
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype021
/dropout_15/dropout/random_uniform/RandomUniform?
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_15/dropout/GreaterEqual/y?
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2!
dropout_15/dropout/GreaterEqual?
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_15/dropout/Cast?
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_15/dropout/Mul_1?
SparseTensor_9/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_9/dense_shape?
SparseSoftmax_7/SparseSoftmaxSparseSoftmaxinputsdropout_15/dropout/Mul_1:z:0#SparseTensor_9/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_7/SparseSoftmax?
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs&SparseSoftmax_7/SparseSoftmax:output:0#SparseTensor_9/dense_shape:output:0dropout_14/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMul?
BiasAdd_7/ReadVariableOpReadVariableOp!biasadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_7/ReadVariableOp?
	BiasAdd_7BiasAdd;SparseTensorDenseMatMul_7/SparseTensorDenseMatMul:product:0 BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_7\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2BiasAdd:output:0BiasAdd_1:output:0BiasAdd_2:output:0BiasAdd_3:output:0BiasAdd_4:output:0BiasAdd_5:output:0BiasAdd_6:output:0BiasAdd_7:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	?@2
concatL
EluEluconcat:output:0*
T0*
_output_shapes
:	?@2
Elub
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsElu:activations:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?@2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^BiasAdd_2/ReadVariableOp^BiasAdd_3/ReadVariableOp^BiasAdd_4/ReadVariableOp^BiasAdd_5/ReadVariableOp^BiasAdd_6/ReadVariableOp^BiasAdd_7/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_10/ReadVariableOp^MatMul_11/ReadVariableOp^MatMul_12/ReadVariableOp^MatMul_13/ReadVariableOp^MatMul_14/ReadVariableOp^MatMul_15/ReadVariableOp^MatMul_16/ReadVariableOp^MatMul_17/ReadVariableOp^MatMul_18/ReadVariableOp^MatMul_19/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_20/ReadVariableOp^MatMul_21/ReadVariableOp^MatMul_22/ReadVariableOp^MatMul_23/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:::::::::::::::::::::::::::::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
BiasAdd_2/ReadVariableOpBiasAdd_2/ReadVariableOp24
BiasAdd_3/ReadVariableOpBiasAdd_3/ReadVariableOp24
BiasAdd_4/ReadVariableOpBiasAdd_4/ReadVariableOp24
BiasAdd_5/ReadVariableOpBiasAdd_5/ReadVariableOp24
BiasAdd_6/ReadVariableOpBiasAdd_6/ReadVariableOp24
BiasAdd_7/ReadVariableOpBiasAdd_7/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp24
MatMul_10/ReadVariableOpMatMul_10/ReadVariableOp24
MatMul_11/ReadVariableOpMatMul_11/ReadVariableOp24
MatMul_12/ReadVariableOpMatMul_12/ReadVariableOp24
MatMul_13/ReadVariableOpMatMul_13/ReadVariableOp24
MatMul_14/ReadVariableOpMatMul_14/ReadVariableOp24
MatMul_15/ReadVariableOpMatMul_15/ReadVariableOp24
MatMul_16/ReadVariableOpMatMul_16/ReadVariableOp24
MatMul_17/ReadVariableOpMatMul_17/ReadVariableOp24
MatMul_18/ReadVariableOpMatMul_18/ReadVariableOp24
MatMul_19/ReadVariableOpMatMul_19/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp24
MatMul_20/ReadVariableOpMatMul_20/ReadVariableOp24
MatMul_21/ReadVariableOpMatMul_21/ReadVariableOp24
MatMul_22/ReadVariableOpMatMul_22/ReadVariableOp24
MatMul_23/ReadVariableOpMatMul_23/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp22
MatMul_8/ReadVariableOpMatMul_8/ReadVariableOp22
MatMul_9/ReadVariableOpMatMul_9/ReadVariableOp:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
]
A__inference_lambda_layer_call_and_return_conditional_losses_12190

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_8464

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constp
dropout/MulMulinputsdropout/Const:output:0*
T0*$
_output_shapes
:??2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?
  ?  2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:??*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:??2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:??2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:??2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:??2

Identity"
identityIdentity:output:0*#
_input_shapes
:??:L H
$
_output_shapes
:??
 
_user_specified_nameinputs
?	
?
:__inference_squeezed_sparse_conversion_layer_call_fn_11160
inputs_0	
inputs_1
identity	

identity_1

identity_2	?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:?????????:?????????:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_84372
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:?????????2

Identityl

Identity_1IdentityPartitionedCall:output:1*
T0*#
_output_shapes
:?????????2

Identity_1c

Identity_2IdentityPartitionedCall:output:2*
T0	*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*=
_input_shapes,
*:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?R
!__inference__traced_restore_12966
file_prefix4
0assignvariableop_graph_attention_sparse_ig_delta?
;assignvariableop_1_graph_attention_sparse_ig_non_exist_edge6
2assignvariableop_2_graph_attention_sparse_kernel_04
0assignvariableop_3_graph_attention_sparse_bias_0@
<assignvariableop_4_graph_attention_sparse_attn_kernel_self_0A
=assignvariableop_5_graph_attention_sparse_attn_kernel_neigh_06
2assignvariableop_6_graph_attention_sparse_kernel_14
0assignvariableop_7_graph_attention_sparse_bias_1@
<assignvariableop_8_graph_attention_sparse_attn_kernel_self_1A
=assignvariableop_9_graph_attention_sparse_attn_kernel_neigh_17
3assignvariableop_10_graph_attention_sparse_kernel_25
1assignvariableop_11_graph_attention_sparse_bias_2A
=assignvariableop_12_graph_attention_sparse_attn_kernel_self_2B
>assignvariableop_13_graph_attention_sparse_attn_kernel_neigh_27
3assignvariableop_14_graph_attention_sparse_kernel_35
1assignvariableop_15_graph_attention_sparse_bias_3A
=assignvariableop_16_graph_attention_sparse_attn_kernel_self_3B
>assignvariableop_17_graph_attention_sparse_attn_kernel_neigh_37
3assignvariableop_18_graph_attention_sparse_kernel_45
1assignvariableop_19_graph_attention_sparse_bias_4A
=assignvariableop_20_graph_attention_sparse_attn_kernel_self_4B
>assignvariableop_21_graph_attention_sparse_attn_kernel_neigh_47
3assignvariableop_22_graph_attention_sparse_kernel_55
1assignvariableop_23_graph_attention_sparse_bias_5A
=assignvariableop_24_graph_attention_sparse_attn_kernel_self_5B
>assignvariableop_25_graph_attention_sparse_attn_kernel_neigh_57
3assignvariableop_26_graph_attention_sparse_kernel_65
1assignvariableop_27_graph_attention_sparse_bias_6A
=assignvariableop_28_graph_attention_sparse_attn_kernel_self_6B
>assignvariableop_29_graph_attention_sparse_attn_kernel_neigh_67
3assignvariableop_30_graph_attention_sparse_kernel_75
1assignvariableop_31_graph_attention_sparse_bias_7A
=assignvariableop_32_graph_attention_sparse_attn_kernel_self_7B
>assignvariableop_33_graph_attention_sparse_attn_kernel_neigh_79
5assignvariableop_34_graph_attention_sparse_1_ig_deltaB
>assignvariableop_35_graph_attention_sparse_1_ig_non_exist_edge9
5assignvariableop_36_graph_attention_sparse_1_kernel_07
3assignvariableop_37_graph_attention_sparse_1_bias_0C
?assignvariableop_38_graph_attention_sparse_1_attn_kernel_self_0D
@assignvariableop_39_graph_attention_sparse_1_attn_kernel_neigh_0!
assignvariableop_40_adam_iter#
assignvariableop_41_adam_beta_1#
assignvariableop_42_adam_beta_2"
assignvariableop_43_adam_decay*
&assignvariableop_44_adam_learning_rate
assignvariableop_45_total
assignvariableop_46_count
assignvariableop_47_total_1
assignvariableop_48_count_1>
:assignvariableop_49_adam_graph_attention_sparse_kernel_0_m<
8assignvariableop_50_adam_graph_attention_sparse_bias_0_mH
Dassignvariableop_51_adam_graph_attention_sparse_attn_kernel_self_0_mI
Eassignvariableop_52_adam_graph_attention_sparse_attn_kernel_neigh_0_m>
:assignvariableop_53_adam_graph_attention_sparse_kernel_1_m<
8assignvariableop_54_adam_graph_attention_sparse_bias_1_mH
Dassignvariableop_55_adam_graph_attention_sparse_attn_kernel_self_1_mI
Eassignvariableop_56_adam_graph_attention_sparse_attn_kernel_neigh_1_m>
:assignvariableop_57_adam_graph_attention_sparse_kernel_2_m<
8assignvariableop_58_adam_graph_attention_sparse_bias_2_mH
Dassignvariableop_59_adam_graph_attention_sparse_attn_kernel_self_2_mI
Eassignvariableop_60_adam_graph_attention_sparse_attn_kernel_neigh_2_m>
:assignvariableop_61_adam_graph_attention_sparse_kernel_3_m<
8assignvariableop_62_adam_graph_attention_sparse_bias_3_mH
Dassignvariableop_63_adam_graph_attention_sparse_attn_kernel_self_3_mI
Eassignvariableop_64_adam_graph_attention_sparse_attn_kernel_neigh_3_m>
:assignvariableop_65_adam_graph_attention_sparse_kernel_4_m<
8assignvariableop_66_adam_graph_attention_sparse_bias_4_mH
Dassignvariableop_67_adam_graph_attention_sparse_attn_kernel_self_4_mI
Eassignvariableop_68_adam_graph_attention_sparse_attn_kernel_neigh_4_m>
:assignvariableop_69_adam_graph_attention_sparse_kernel_5_m<
8assignvariableop_70_adam_graph_attention_sparse_bias_5_mH
Dassignvariableop_71_adam_graph_attention_sparse_attn_kernel_self_5_mI
Eassignvariableop_72_adam_graph_attention_sparse_attn_kernel_neigh_5_m>
:assignvariableop_73_adam_graph_attention_sparse_kernel_6_m<
8assignvariableop_74_adam_graph_attention_sparse_bias_6_mH
Dassignvariableop_75_adam_graph_attention_sparse_attn_kernel_self_6_mI
Eassignvariableop_76_adam_graph_attention_sparse_attn_kernel_neigh_6_m>
:assignvariableop_77_adam_graph_attention_sparse_kernel_7_m<
8assignvariableop_78_adam_graph_attention_sparse_bias_7_mH
Dassignvariableop_79_adam_graph_attention_sparse_attn_kernel_self_7_mI
Eassignvariableop_80_adam_graph_attention_sparse_attn_kernel_neigh_7_m@
<assignvariableop_81_adam_graph_attention_sparse_1_kernel_0_m>
:assignvariableop_82_adam_graph_attention_sparse_1_bias_0_mJ
Fassignvariableop_83_adam_graph_attention_sparse_1_attn_kernel_self_0_mK
Gassignvariableop_84_adam_graph_attention_sparse_1_attn_kernel_neigh_0_m>
:assignvariableop_85_adam_graph_attention_sparse_kernel_0_v<
8assignvariableop_86_adam_graph_attention_sparse_bias_0_vH
Dassignvariableop_87_adam_graph_attention_sparse_attn_kernel_self_0_vI
Eassignvariableop_88_adam_graph_attention_sparse_attn_kernel_neigh_0_v>
:assignvariableop_89_adam_graph_attention_sparse_kernel_1_v<
8assignvariableop_90_adam_graph_attention_sparse_bias_1_vH
Dassignvariableop_91_adam_graph_attention_sparse_attn_kernel_self_1_vI
Eassignvariableop_92_adam_graph_attention_sparse_attn_kernel_neigh_1_v>
:assignvariableop_93_adam_graph_attention_sparse_kernel_2_v<
8assignvariableop_94_adam_graph_attention_sparse_bias_2_vH
Dassignvariableop_95_adam_graph_attention_sparse_attn_kernel_self_2_vI
Eassignvariableop_96_adam_graph_attention_sparse_attn_kernel_neigh_2_v>
:assignvariableop_97_adam_graph_attention_sparse_kernel_3_v<
8assignvariableop_98_adam_graph_attention_sparse_bias_3_vH
Dassignvariableop_99_adam_graph_attention_sparse_attn_kernel_self_3_vJ
Fassignvariableop_100_adam_graph_attention_sparse_attn_kernel_neigh_3_v?
;assignvariableop_101_adam_graph_attention_sparse_kernel_4_v=
9assignvariableop_102_adam_graph_attention_sparse_bias_4_vI
Eassignvariableop_103_adam_graph_attention_sparse_attn_kernel_self_4_vJ
Fassignvariableop_104_adam_graph_attention_sparse_attn_kernel_neigh_4_v?
;assignvariableop_105_adam_graph_attention_sparse_kernel_5_v=
9assignvariableop_106_adam_graph_attention_sparse_bias_5_vI
Eassignvariableop_107_adam_graph_attention_sparse_attn_kernel_self_5_vJ
Fassignvariableop_108_adam_graph_attention_sparse_attn_kernel_neigh_5_v?
;assignvariableop_109_adam_graph_attention_sparse_kernel_6_v=
9assignvariableop_110_adam_graph_attention_sparse_bias_6_vI
Eassignvariableop_111_adam_graph_attention_sparse_attn_kernel_self_6_vJ
Fassignvariableop_112_adam_graph_attention_sparse_attn_kernel_neigh_6_v?
;assignvariableop_113_adam_graph_attention_sparse_kernel_7_v=
9assignvariableop_114_adam_graph_attention_sparse_bias_7_vI
Eassignvariableop_115_adam_graph_attention_sparse_attn_kernel_self_7_vJ
Fassignvariableop_116_adam_graph_attention_sparse_attn_kernel_neigh_7_vA
=assignvariableop_117_adam_graph_attention_sparse_1_kernel_0_v?
;assignvariableop_118_adam_graph_attention_sparse_1_bias_0_vK
Gassignvariableop_119_adam_graph_attention_sparse_1_attn_kernel_self_0_vL
Hassignvariableop_120_adam_graph_attention_sparse_1_attn_kernel_neigh_0_v
identity_122??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?K
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?J
value?JB?JzB8layer_with_weights-0/ig_delta/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/ig_non_exist_edge/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_0/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_0/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_0/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_0/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_1/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_1/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_1/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_2/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_2/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_2/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_3/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_3/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_3/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_3/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_4/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_4/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_4/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_4/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_5/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_5/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_5/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_5/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_6/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_6/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_6/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_6/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_7/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_7/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_7/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_7/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/ig_delta/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/ig_non_exist_edge/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/kernel_0/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/bias_0/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/attn_kernel_self_0/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/attn_kernel_neigh_0/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-1/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-1/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?
value?B?zB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes~
|2z	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp0assignvariableop_graph_attention_sparse_ig_deltaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp;assignvariableop_1_graph_attention_sparse_ig_non_exist_edgeIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp2assignvariableop_2_graph_attention_sparse_kernel_0Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp0assignvariableop_3_graph_attention_sparse_bias_0Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp<assignvariableop_4_graph_attention_sparse_attn_kernel_self_0Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp=assignvariableop_5_graph_attention_sparse_attn_kernel_neigh_0Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp2assignvariableop_6_graph_attention_sparse_kernel_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp0assignvariableop_7_graph_attention_sparse_bias_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp<assignvariableop_8_graph_attention_sparse_attn_kernel_self_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp=assignvariableop_9_graph_attention_sparse_attn_kernel_neigh_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_graph_attention_sparse_kernel_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_graph_attention_sparse_bias_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp=assignvariableop_12_graph_attention_sparse_attn_kernel_self_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp>assignvariableop_13_graph_attention_sparse_attn_kernel_neigh_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp3assignvariableop_14_graph_attention_sparse_kernel_3Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_graph_attention_sparse_bias_3Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp=assignvariableop_16_graph_attention_sparse_attn_kernel_self_3Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp>assignvariableop_17_graph_attention_sparse_attn_kernel_neigh_3Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp3assignvariableop_18_graph_attention_sparse_kernel_4Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_graph_attention_sparse_bias_4Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp=assignvariableop_20_graph_attention_sparse_attn_kernel_self_4Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp>assignvariableop_21_graph_attention_sparse_attn_kernel_neigh_4Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_graph_attention_sparse_kernel_5Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_graph_attention_sparse_bias_5Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp=assignvariableop_24_graph_attention_sparse_attn_kernel_self_5Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp>assignvariableop_25_graph_attention_sparse_attn_kernel_neigh_5Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_graph_attention_sparse_kernel_6Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_graph_attention_sparse_bias_6Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp=assignvariableop_28_graph_attention_sparse_attn_kernel_self_6Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp>assignvariableop_29_graph_attention_sparse_attn_kernel_neigh_6Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp3assignvariableop_30_graph_attention_sparse_kernel_7Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp1assignvariableop_31_graph_attention_sparse_bias_7Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp=assignvariableop_32_graph_attention_sparse_attn_kernel_self_7Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp>assignvariableop_33_graph_attention_sparse_attn_kernel_neigh_7Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp5assignvariableop_34_graph_attention_sparse_1_ig_deltaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp>assignvariableop_35_graph_attention_sparse_1_ig_non_exist_edgeIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp5assignvariableop_36_graph_attention_sparse_1_kernel_0Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp3assignvariableop_37_graph_attention_sparse_1_bias_0Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp?assignvariableop_38_graph_attention_sparse_1_attn_kernel_self_0Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp@assignvariableop_39_graph_attention_sparse_1_attn_kernel_neigh_0Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_iterIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_beta_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_beta_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_decayIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_learning_rateIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp:assignvariableop_49_adam_graph_attention_sparse_kernel_0_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp8assignvariableop_50_adam_graph_attention_sparse_bias_0_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpDassignvariableop_51_adam_graph_attention_sparse_attn_kernel_self_0_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpEassignvariableop_52_adam_graph_attention_sparse_attn_kernel_neigh_0_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp:assignvariableop_53_adam_graph_attention_sparse_kernel_1_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp8assignvariableop_54_adam_graph_attention_sparse_bias_1_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpDassignvariableop_55_adam_graph_attention_sparse_attn_kernel_self_1_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpEassignvariableop_56_adam_graph_attention_sparse_attn_kernel_neigh_1_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp:assignvariableop_57_adam_graph_attention_sparse_kernel_2_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp8assignvariableop_58_adam_graph_attention_sparse_bias_2_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpDassignvariableop_59_adam_graph_attention_sparse_attn_kernel_self_2_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpEassignvariableop_60_adam_graph_attention_sparse_attn_kernel_neigh_2_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp:assignvariableop_61_adam_graph_attention_sparse_kernel_3_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp8assignvariableop_62_adam_graph_attention_sparse_bias_3_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpDassignvariableop_63_adam_graph_attention_sparse_attn_kernel_self_3_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpEassignvariableop_64_adam_graph_attention_sparse_attn_kernel_neigh_3_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp:assignvariableop_65_adam_graph_attention_sparse_kernel_4_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp8assignvariableop_66_adam_graph_attention_sparse_bias_4_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpDassignvariableop_67_adam_graph_attention_sparse_attn_kernel_self_4_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpEassignvariableop_68_adam_graph_attention_sparse_attn_kernel_neigh_4_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp:assignvariableop_69_adam_graph_attention_sparse_kernel_5_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp8assignvariableop_70_adam_graph_attention_sparse_bias_5_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpDassignvariableop_71_adam_graph_attention_sparse_attn_kernel_self_5_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpEassignvariableop_72_adam_graph_attention_sparse_attn_kernel_neigh_5_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp:assignvariableop_73_adam_graph_attention_sparse_kernel_6_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp8assignvariableop_74_adam_graph_attention_sparse_bias_6_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpDassignvariableop_75_adam_graph_attention_sparse_attn_kernel_self_6_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOpEassignvariableop_76_adam_graph_attention_sparse_attn_kernel_neigh_6_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp:assignvariableop_77_adam_graph_attention_sparse_kernel_7_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp8assignvariableop_78_adam_graph_attention_sparse_bias_7_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOpDassignvariableop_79_adam_graph_attention_sparse_attn_kernel_self_7_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOpEassignvariableop_80_adam_graph_attention_sparse_attn_kernel_neigh_7_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp<assignvariableop_81_adam_graph_attention_sparse_1_kernel_0_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp:assignvariableop_82_adam_graph_attention_sparse_1_bias_0_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpFassignvariableop_83_adam_graph_attention_sparse_1_attn_kernel_self_0_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOpGassignvariableop_84_adam_graph_attention_sparse_1_attn_kernel_neigh_0_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp:assignvariableop_85_adam_graph_attention_sparse_kernel_0_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp8assignvariableop_86_adam_graph_attention_sparse_bias_0_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpDassignvariableop_87_adam_graph_attention_sparse_attn_kernel_self_0_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpEassignvariableop_88_adam_graph_attention_sparse_attn_kernel_neigh_0_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp:assignvariableop_89_adam_graph_attention_sparse_kernel_1_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp8assignvariableop_90_adam_graph_attention_sparse_bias_1_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOpDassignvariableop_91_adam_graph_attention_sparse_attn_kernel_self_1_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOpEassignvariableop_92_adam_graph_attention_sparse_attn_kernel_neigh_1_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp:assignvariableop_93_adam_graph_attention_sparse_kernel_2_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp8assignvariableop_94_adam_graph_attention_sparse_bias_2_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpDassignvariableop_95_adam_graph_attention_sparse_attn_kernel_self_2_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpEassignvariableop_96_adam_graph_attention_sparse_attn_kernel_neigh_2_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp:assignvariableop_97_adam_graph_attention_sparse_kernel_3_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp8assignvariableop_98_adam_graph_attention_sparse_bias_3_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOpDassignvariableop_99_adam_graph_attention_sparse_attn_kernel_self_3_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOpFassignvariableop_100_adam_graph_attention_sparse_attn_kernel_neigh_3_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp;assignvariableop_101_adam_graph_attention_sparse_kernel_4_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp9assignvariableop_102_adam_graph_attention_sparse_bias_4_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOpEassignvariableop_103_adam_graph_attention_sparse_attn_kernel_self_4_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOpFassignvariableop_104_adam_graph_attention_sparse_attn_kernel_neigh_4_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp;assignvariableop_105_adam_graph_attention_sparse_kernel_5_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp9assignvariableop_106_adam_graph_attention_sparse_bias_5_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOpEassignvariableop_107_adam_graph_attention_sparse_attn_kernel_self_5_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOpFassignvariableop_108_adam_graph_attention_sparse_attn_kernel_neigh_5_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp;assignvariableop_109_adam_graph_attention_sparse_kernel_6_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp9assignvariableop_110_adam_graph_attention_sparse_bias_6_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOpEassignvariableop_111_adam_graph_attention_sparse_attn_kernel_self_6_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOpFassignvariableop_112_adam_graph_attention_sparse_attn_kernel_neigh_6_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp;assignvariableop_113_adam_graph_attention_sparse_kernel_7_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp9assignvariableop_114_adam_graph_attention_sparse_bias_7_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOpEassignvariableop_115_adam_graph_attention_sparse_attn_kernel_self_7_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOpFassignvariableop_116_adam_graph_attention_sparse_attn_kernel_neigh_7_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp=assignvariableop_117_adam_graph_attention_sparse_1_kernel_0_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp;assignvariableop_118_adam_graph_attention_sparse_1_bias_0_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOpGassignvariableop_119_adam_graph_attention_sparse_1_attn_kernel_self_0_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOpHassignvariableop_120_adam_graph_attention_sparse_1_attn_kernel_neigh_0_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_121Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_121?
Identity_122IdentityIdentity_121:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_122"%
identity_122Identity_122:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202*
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
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
E
)__inference_dropout_1_layer_call_fn_12029

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_94082
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
??
?
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_8887

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource%
!biasadd_1_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource$
 matmul_8_readvariableop_resource%
!biasadd_2_readvariableop_resource$
 matmul_9_readvariableop_resource%
!matmul_10_readvariableop_resource%
!matmul_11_readvariableop_resource%
!biasadd_3_readvariableop_resource%
!matmul_12_readvariableop_resource%
!matmul_13_readvariableop_resource%
!matmul_14_readvariableop_resource%
!biasadd_4_readvariableop_resource%
!matmul_15_readvariableop_resource%
!matmul_16_readvariableop_resource%
!matmul_17_readvariableop_resource%
!biasadd_5_readvariableop_resource%
!matmul_18_readvariableop_resource%
!matmul_19_readvariableop_resource%
!matmul_20_readvariableop_resource%
!biasadd_6_readvariableop_resource%
!matmul_21_readvariableop_resource%
!matmul_22_readvariableop_resource%
!matmul_23_readvariableop_resource%
!biasadd_7_readvariableop_resource
identity??BiasAdd/ReadVariableOp?BiasAdd_1/ReadVariableOp?BiasAdd_2/ReadVariableOp?BiasAdd_3/ReadVariableOp?BiasAdd_4/ReadVariableOp?BiasAdd_5/ReadVariableOp?BiasAdd_6/ReadVariableOp?BiasAdd_7/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_10/ReadVariableOp?MatMul_11/ReadVariableOp?MatMul_12/ReadVariableOp?MatMul_13/ReadVariableOp?MatMul_14/ReadVariableOp?MatMul_15/ReadVariableOp?MatMul_16/ReadVariableOp?MatMul_17/ReadVariableOp?MatMul_18/ReadVariableOp?MatMul_19/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_20/ReadVariableOp?MatMul_21/ReadVariableOp?MatMul_22/ReadVariableOp?MatMul_23/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?MatMul_8/ReadVariableOp?MatMul_9/ReadVariableOpg
SqueezeSqueezeinputs*
T0* 
_output_shapes
:
??*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulMatMul:product:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul#leaky_re_lu/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputs_1dropout_1/dropout/Mul_1:z:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_3/ReadVariableOp{
MatMul_3MatMulSqueeze:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_3?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_4/ReadVariableOp}
MatMul_4MatMulMatMul_3:product:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp}
MatMul_5MatMulMatMul_3:product:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_5u
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_2/shapeu
	Reshape_2ReshapeMatMul_4:product:0Reshape_2/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_2:output:0strided_slice_2:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_2u
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_3/shapeu
	Reshape_3ReshapeMatMul_5:product:0Reshape_3/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputs_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_3:output:0strided_slice_3:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_3o
add_1AddV2GatherV2_2:output:0GatherV2_3:output:0*
T0*#
_output_shapes
:?????????2
add_1o
leaky_re_lu_1/LeakyRelu	LeakyRelu	add_1:z:0*#
_output_shapes
:?????????2
leaky_re_lu_1/LeakyReluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulMatMul_3:product:0 dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_2/dropout/Mul_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const?
dropout_3/dropout/MulMul%leaky_re_lu_1/LeakyRelu:activations:0 dropout_3/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_3/dropout/Mul_1?
SparseTensor_3/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_3/dense_shape?
SparseSoftmax_1/SparseSoftmaxSparseSoftmaxinputs_1dropout_3/dropout/Mul_1:z:0#SparseTensor_3/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_1/SparseSoftmax?
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_1/SparseSoftmax:output:0#SparseTensor_3/dense_shape:output:0dropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_1/SparseTensorDenseMatMul?
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_1/ReadVariableOp?
	BiasAdd_1BiasAdd;SparseTensorDenseMatMul_1/SparseTensorDenseMatMul:product:0 BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_1?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_6/ReadVariableOp{
MatMul_6MatMulSqueeze:output:0MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_7/ReadVariableOp}
MatMul_7MatMulMatMul_6:product:0MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_7?
MatMul_8/ReadVariableOpReadVariableOp matmul_8_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_8/ReadVariableOp}
MatMul_8MatMulMatMul_6:product:0MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_8u
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_4/shapeu
	Reshape_4ReshapeMatMul_7:product:0Reshape_4/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_4
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputs_1strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_4:output:0strided_slice_4:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_4u
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_5/shapeu
	Reshape_5ReshapeMatMul_8:product:0Reshape_5/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_5
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceinputs_1strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_5:output:0strided_slice_5:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_5o
add_2AddV2GatherV2_4:output:0GatherV2_5:output:0*
T0*#
_output_shapes
:?????????2
add_2o
leaky_re_lu_2/LeakyRelu	LeakyRelu	add_2:z:0*#
_output_shapes
:?????????2
leaky_re_lu_2/LeakyReluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulMatMul_6:product:0 dropout_4/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_4/dropout/Mul_1w
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul%leaky_re_lu_2/LeakyRelu:activations:0 dropout_5/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_5/dropout/Mul?
dropout_5/dropout/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_5/dropout/Mul_1?
SparseTensor_4/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_4/dense_shape?
SparseSoftmax_2/SparseSoftmaxSparseSoftmaxinputs_1dropout_5/dropout/Mul_1:z:0#SparseTensor_4/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_2/SparseSoftmax?
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_2/SparseSoftmax:output:0#SparseTensor_4/dense_shape:output:0dropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_2/SparseTensorDenseMatMul?
BiasAdd_2/ReadVariableOpReadVariableOp!biasadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_2/ReadVariableOp?
	BiasAdd_2BiasAdd;SparseTensorDenseMatMul_2/SparseTensorDenseMatMul:product:0 BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_2?
MatMul_9/ReadVariableOpReadVariableOp matmul_9_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_9/ReadVariableOp{
MatMul_9MatMulSqueeze:output:0MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_9?
MatMul_10/ReadVariableOpReadVariableOp!matmul_10_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_10/ReadVariableOp?
	MatMul_10MatMulMatMul_9:product:0 MatMul_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_10?
MatMul_11/ReadVariableOpReadVariableOp!matmul_11_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_11/ReadVariableOp?
	MatMul_11MatMulMatMul_9:product:0 MatMul_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_11u
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_6/shapev
	Reshape_6ReshapeMatMul_10:product:0Reshape_6/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceinputs_1strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_6:output:0strided_slice_6:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_6u
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_7/shapev
	Reshape_7ReshapeMatMul_11:product:0Reshape_7/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceinputs_1strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_7:output:0strided_slice_7:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_7o
add_3AddV2GatherV2_6:output:0GatherV2_7:output:0*
T0*#
_output_shapes
:?????????2
add_3o
leaky_re_lu_3/LeakyRelu	LeakyRelu	add_3:z:0*#
_output_shapes
:?????????2
leaky_re_lu_3/LeakyReluw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_6/dropout/Const?
dropout_6/dropout/MulMulMatMul_9:product:0 dropout_6/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_6/dropout/Mul?
dropout_6/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_6/dropout/Mul_1w
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_7/dropout/Const?
dropout_7/dropout/MulMul%leaky_re_lu_3/LeakyRelu:activations:0 dropout_7/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_7/dropout/Mul?
dropout_7/dropout/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_7/dropout/Mul_1?
SparseTensor_5/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_5/dense_shape?
SparseSoftmax_3/SparseSoftmaxSparseSoftmaxinputs_1dropout_7/dropout/Mul_1:z:0#SparseTensor_5/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_3/SparseSoftmax?
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_3/SparseSoftmax:output:0#SparseTensor_5/dense_shape:output:0dropout_6/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_3/SparseTensorDenseMatMul?
BiasAdd_3/ReadVariableOpReadVariableOp!biasadd_3_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_3/ReadVariableOp?
	BiasAdd_3BiasAdd;SparseTensorDenseMatMul_3/SparseTensorDenseMatMul:product:0 BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_3?
MatMul_12/ReadVariableOpReadVariableOp!matmul_12_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_12/ReadVariableOp~
	MatMul_12MatMulSqueeze:output:0 MatMul_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_12?
MatMul_13/ReadVariableOpReadVariableOp!matmul_13_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_13/ReadVariableOp?
	MatMul_13MatMulMatMul_12:product:0 MatMul_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_13?
MatMul_14/ReadVariableOpReadVariableOp!matmul_14_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_14/ReadVariableOp?
	MatMul_14MatMulMatMul_12:product:0 MatMul_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_14u
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_8/shapev
	Reshape_8ReshapeMatMul_13:product:0Reshape_8/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceinputs_1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8d
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axis?

GatherV2_8GatherV2Reshape_8:output:0strided_slice_8:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_8u
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_9/shapev
	Reshape_9ReshapeMatMul_14:product:0Reshape_9/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_9
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceinputs_1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9d
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axis?

GatherV2_9GatherV2Reshape_9:output:0strided_slice_9:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_9o
add_4AddV2GatherV2_8:output:0GatherV2_9:output:0*
T0*#
_output_shapes
:?????????2
add_4o
leaky_re_lu_4/LeakyRelu	LeakyRelu	add_4:z:0*#
_output_shapes
:?????????2
leaky_re_lu_4/LeakyReluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_8/dropout/Const?
dropout_8/dropout/MulMulMatMul_12:product:0 dropout_8/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_8/dropout/Mul_1w
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul%leaky_re_lu_4/LeakyRelu:activations:0 dropout_9/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_9/dropout/Mul_1?
SparseTensor_6/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_6/dense_shape?
SparseSoftmax_4/SparseSoftmaxSparseSoftmaxinputs_1dropout_9/dropout/Mul_1:z:0#SparseTensor_6/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_4/SparseSoftmax?
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_4/SparseSoftmax:output:0#SparseTensor_6/dense_shape:output:0dropout_8/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_4/SparseTensorDenseMatMul?
BiasAdd_4/ReadVariableOpReadVariableOp!biasadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_4/ReadVariableOp?
	BiasAdd_4BiasAdd;SparseTensorDenseMatMul_4/SparseTensorDenseMatMul:product:0 BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_4?
MatMul_15/ReadVariableOpReadVariableOp!matmul_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_15/ReadVariableOp~
	MatMul_15MatMulSqueeze:output:0 MatMul_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_15?
MatMul_16/ReadVariableOpReadVariableOp!matmul_16_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_16/ReadVariableOp?
	MatMul_16MatMulMatMul_15:product:0 MatMul_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_16?
MatMul_17/ReadVariableOpReadVariableOp!matmul_17_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_17/ReadVariableOp?
	MatMul_17MatMulMatMul_15:product:0 MatMul_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_17w
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_10/shapey

Reshape_10ReshapeMatMul_16:product:0Reshape_10/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_10?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceinputs_1strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10f
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axis?
GatherV2_10GatherV2Reshape_10:output:0strided_slice_10:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_10w
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_11/shapey

Reshape_11ReshapeMatMul_17:product:0Reshape_11/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_11?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceinputs_1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11f
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axis?
GatherV2_11GatherV2Reshape_11:output:0strided_slice_11:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_11q
add_5AddV2GatherV2_10:output:0GatherV2_11:output:0*
T0*#
_output_shapes
:?????????2
add_5o
leaky_re_lu_5/LeakyRelu	LeakyRelu	add_5:z:0*#
_output_shapes
:?????????2
leaky_re_lu_5/LeakyReluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Const?
dropout_10/dropout/MulMulMatMul_15:product:0!dropout_10/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_10/dropout/Mul?
dropout_10/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform?
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/y?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_10/dropout/Mul_1y
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Const?
dropout_11/dropout/MulMul%leaky_re_lu_5/LeakyRelu:activations:0!dropout_11/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_11/dropout/Mul?
dropout_11/dropout/ShapeShape%leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform?
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/y?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2!
dropout_11/dropout/GreaterEqual?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_11/dropout/Cast?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_11/dropout/Mul_1?
SparseTensor_7/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_7/dense_shape?
SparseSoftmax_5/SparseSoftmaxSparseSoftmaxinputs_1dropout_11/dropout/Mul_1:z:0#SparseTensor_7/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_5/SparseSoftmax?
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_5/SparseSoftmax:output:0#SparseTensor_7/dense_shape:output:0dropout_10/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_5/SparseTensorDenseMatMul?
BiasAdd_5/ReadVariableOpReadVariableOp!biasadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_5/ReadVariableOp?
	BiasAdd_5BiasAdd;SparseTensorDenseMatMul_5/SparseTensorDenseMatMul:product:0 BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_5?
MatMul_18/ReadVariableOpReadVariableOp!matmul_18_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_18/ReadVariableOp~
	MatMul_18MatMulSqueeze:output:0 MatMul_18/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_18?
MatMul_19/ReadVariableOpReadVariableOp!matmul_19_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_19/ReadVariableOp?
	MatMul_19MatMulMatMul_18:product:0 MatMul_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_19?
MatMul_20/ReadVariableOpReadVariableOp!matmul_20_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_20/ReadVariableOp?
	MatMul_20MatMulMatMul_18:product:0 MatMul_20/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_20w
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_12/shapey

Reshape_12ReshapeMatMul_19:product:0Reshape_12/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_12?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSliceinputs_1strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12f
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axis?
GatherV2_12GatherV2Reshape_12:output:0strided_slice_12:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_12w
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_13/shapey

Reshape_13ReshapeMatMul_20:product:0Reshape_13/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_13?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSliceinputs_1strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13f
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axis?
GatherV2_13GatherV2Reshape_13:output:0strided_slice_13:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_13q
add_6AddV2GatherV2_12:output:0GatherV2_13:output:0*
T0*#
_output_shapes
:?????????2
add_6o
leaky_re_lu_6/LeakyRelu	LeakyRelu	add_6:z:0*#
_output_shapes
:?????????2
leaky_re_lu_6/LeakyReluy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_12/dropout/Const?
dropout_12/dropout/MulMulMatMul_18:product:0!dropout_12/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_12/dropout/Mul?
dropout_12/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_12/dropout/Shape?
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform?
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_12/dropout/GreaterEqual/y?
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2!
dropout_12/dropout/GreaterEqual?
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_12/dropout/Cast?
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_12/dropout/Mul_1y
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Const?
dropout_13/dropout/MulMul%leaky_re_lu_6/LeakyRelu:activations:0!dropout_13/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_13/dropout/Mul?
dropout_13/dropout/ShapeShape%leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform?
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/y?
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2!
dropout_13/dropout/GreaterEqual?
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_13/dropout/Cast?
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_13/dropout/Mul_1?
SparseTensor_8/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_8/dense_shape?
SparseSoftmax_6/SparseSoftmaxSparseSoftmaxinputs_1dropout_13/dropout/Mul_1:z:0#SparseTensor_8/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_6/SparseSoftmax?
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_6/SparseSoftmax:output:0#SparseTensor_8/dense_shape:output:0dropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_6/SparseTensorDenseMatMul?
BiasAdd_6/ReadVariableOpReadVariableOp!biasadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_6/ReadVariableOp?
	BiasAdd_6BiasAdd;SparseTensorDenseMatMul_6/SparseTensorDenseMatMul:product:0 BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_6?
MatMul_21/ReadVariableOpReadVariableOp!matmul_21_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul_21/ReadVariableOp~
	MatMul_21MatMulSqueeze:output:0 MatMul_21/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_21?
MatMul_22/ReadVariableOpReadVariableOp!matmul_22_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_22/ReadVariableOp?
	MatMul_22MatMulMatMul_21:product:0 MatMul_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_22?
MatMul_23/ReadVariableOpReadVariableOp!matmul_23_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_23/ReadVariableOp?
	MatMul_23MatMulMatMul_21:product:0 MatMul_23/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	MatMul_23w
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_14/shapey

Reshape_14ReshapeMatMul_22:product:0Reshape_14/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_14?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSliceinputs_1strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14f
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axis?
GatherV2_14GatherV2Reshape_14:output:0strided_slice_14:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_14w
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_15/shapey

Reshape_15ReshapeMatMul_23:product:0Reshape_15/shape:output:0*
T0*
_output_shapes	
:?2

Reshape_15?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSliceinputs_1strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15f
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axis?
GatherV2_15GatherV2Reshape_15:output:0strided_slice_15:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
GatherV2_15q
add_7AddV2GatherV2_14:output:0GatherV2_15:output:0*
T0*#
_output_shapes
:?????????2
add_7o
leaky_re_lu_7/LeakyRelu	LeakyRelu	add_7:z:0*#
_output_shapes
:?????????2
leaky_re_lu_7/LeakyReluy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_14/dropout/Const?
dropout_14/dropout/MulMulMatMul_21:product:0!dropout_14/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout_14/dropout/Mul?
dropout_14/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout_14/dropout/Shape?
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype021
/dropout_14/dropout/random_uniform/RandomUniform?
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_14/dropout/GreaterEqual/y?
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2!
dropout_14/dropout/GreaterEqual?
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout_14/dropout/Cast?
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout_14/dropout/Mul_1y
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_15/dropout/Const?
dropout_15/dropout/MulMul%leaky_re_lu_7/LeakyRelu:activations:0!dropout_15/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_15/dropout/Mul?
dropout_15/dropout/ShapeShape%leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape?
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype021
/dropout_15/dropout/random_uniform/RandomUniform?
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_15/dropout/GreaterEqual/y?
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2!
dropout_15/dropout/GreaterEqual?
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_15/dropout/Cast?
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_15/dropout/Mul_1?
SparseTensor_9/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_9/dense_shape?
SparseSoftmax_7/SparseSoftmaxSparseSoftmaxinputs_1dropout_15/dropout/Mul_1:z:0#SparseTensor_9/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax_7/SparseSoftmax?
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1&SparseSoftmax_7/SparseSoftmax:output:0#SparseTensor_9/dense_shape:output:0dropout_14/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?23
1SparseTensorDenseMatMul_7/SparseTensorDenseMatMul?
BiasAdd_7/ReadVariableOpReadVariableOp!biasadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd_7/ReadVariableOp?
	BiasAdd_7BiasAdd;SparseTensorDenseMatMul_7/SparseTensorDenseMatMul:product:0 BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
	BiasAdd_7\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2BiasAdd:output:0BiasAdd_1:output:0BiasAdd_2:output:0BiasAdd_3:output:0BiasAdd_4:output:0BiasAdd_5:output:0BiasAdd_6:output:0BiasAdd_7:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	?@2
concatL
EluEluconcat:output:0*
T0*
_output_shapes
:	?@2
Elub
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsElu:activations:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?@2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^BiasAdd_2/ReadVariableOp^BiasAdd_3/ReadVariableOp^BiasAdd_4/ReadVariableOp^BiasAdd_5/ReadVariableOp^BiasAdd_6/ReadVariableOp^BiasAdd_7/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_10/ReadVariableOp^MatMul_11/ReadVariableOp^MatMul_12/ReadVariableOp^MatMul_13/ReadVariableOp^MatMul_14/ReadVariableOp^MatMul_15/ReadVariableOp^MatMul_16/ReadVariableOp^MatMul_17/ReadVariableOp^MatMul_18/ReadVariableOp^MatMul_19/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_20/ReadVariableOp^MatMul_21/ReadVariableOp^MatMul_22/ReadVariableOp^MatMul_23/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:::::::::::::::::::::::::::::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
BiasAdd_2/ReadVariableOpBiasAdd_2/ReadVariableOp24
BiasAdd_3/ReadVariableOpBiasAdd_3/ReadVariableOp24
BiasAdd_4/ReadVariableOpBiasAdd_4/ReadVariableOp24
BiasAdd_5/ReadVariableOpBiasAdd_5/ReadVariableOp24
BiasAdd_6/ReadVariableOpBiasAdd_6/ReadVariableOp24
BiasAdd_7/ReadVariableOpBiasAdd_7/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp24
MatMul_10/ReadVariableOpMatMul_10/ReadVariableOp24
MatMul_11/ReadVariableOpMatMul_11/ReadVariableOp24
MatMul_12/ReadVariableOpMatMul_12/ReadVariableOp24
MatMul_13/ReadVariableOpMatMul_13/ReadVariableOp24
MatMul_14/ReadVariableOpMatMul_14/ReadVariableOp24
MatMul_15/ReadVariableOpMatMul_15/ReadVariableOp24
MatMul_16/ReadVariableOpMatMul_16/ReadVariableOp24
MatMul_17/ReadVariableOpMatMul_17/ReadVariableOp24
MatMul_18/ReadVariableOpMatMul_18/ReadVariableOp24
MatMul_19/ReadVariableOpMatMul_19/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp24
MatMul_20/ReadVariableOpMatMul_20/ReadVariableOp24
MatMul_21/ReadVariableOpMatMul_21/ReadVariableOp24
MatMul_22/ReadVariableOpMatMul_22/ReadVariableOp24
MatMul_23/ReadVariableOpMatMul_23/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp22
MatMul_8/ReadVariableOpMatMul_8/ReadVariableOp22
MatMul_9/ReadVariableOpMatMul_9/ReadVariableOp:L H
$
_output_shapes
:??
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_11130

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:??2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:??2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:??:L H
$
_output_shapes
:??
 
_user_specified_nameinputs
??
?D
__inference__traced_save_12593
file_prefix>
:savev2_graph_attention_sparse_ig_delta_read_readvariableopG
Csavev2_graph_attention_sparse_ig_non_exist_edge_read_readvariableop>
:savev2_graph_attention_sparse_kernel_0_read_readvariableop<
8savev2_graph_attention_sparse_bias_0_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_0_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_0_read_readvariableop>
:savev2_graph_attention_sparse_kernel_1_read_readvariableop<
8savev2_graph_attention_sparse_bias_1_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_1_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_1_read_readvariableop>
:savev2_graph_attention_sparse_kernel_2_read_readvariableop<
8savev2_graph_attention_sparse_bias_2_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_2_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_2_read_readvariableop>
:savev2_graph_attention_sparse_kernel_3_read_readvariableop<
8savev2_graph_attention_sparse_bias_3_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_3_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_3_read_readvariableop>
:savev2_graph_attention_sparse_kernel_4_read_readvariableop<
8savev2_graph_attention_sparse_bias_4_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_4_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_4_read_readvariableop>
:savev2_graph_attention_sparse_kernel_5_read_readvariableop<
8savev2_graph_attention_sparse_bias_5_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_5_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_5_read_readvariableop>
:savev2_graph_attention_sparse_kernel_6_read_readvariableop<
8savev2_graph_attention_sparse_bias_6_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_6_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_6_read_readvariableop>
:savev2_graph_attention_sparse_kernel_7_read_readvariableop<
8savev2_graph_attention_sparse_bias_7_read_readvariableopH
Dsavev2_graph_attention_sparse_attn_kernel_self_7_read_readvariableopI
Esavev2_graph_attention_sparse_attn_kernel_neigh_7_read_readvariableop@
<savev2_graph_attention_sparse_1_ig_delta_read_readvariableopI
Esavev2_graph_attention_sparse_1_ig_non_exist_edge_read_readvariableop@
<savev2_graph_attention_sparse_1_kernel_0_read_readvariableop>
:savev2_graph_attention_sparse_1_bias_0_read_readvariableopJ
Fsavev2_graph_attention_sparse_1_attn_kernel_self_0_read_readvariableopK
Gsavev2_graph_attention_sparse_1_attn_kernel_neigh_0_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_0_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_0_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_0_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_0_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_1_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_1_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_1_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_1_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_2_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_2_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_2_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_2_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_3_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_3_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_3_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_3_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_4_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_4_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_4_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_4_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_5_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_5_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_5_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_5_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_6_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_6_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_6_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_6_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_7_m_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_7_m_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_7_m_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_7_m_read_readvariableopG
Csavev2_adam_graph_attention_sparse_1_kernel_0_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_1_bias_0_m_read_readvariableopQ
Msavev2_adam_graph_attention_sparse_1_attn_kernel_self_0_m_read_readvariableopR
Nsavev2_adam_graph_attention_sparse_1_attn_kernel_neigh_0_m_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_0_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_0_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_0_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_0_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_1_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_1_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_1_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_1_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_2_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_2_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_2_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_2_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_3_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_3_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_3_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_3_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_4_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_4_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_4_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_4_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_5_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_5_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_5_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_5_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_6_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_6_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_6_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_6_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_kernel_7_v_read_readvariableopC
?savev2_adam_graph_attention_sparse_bias_7_v_read_readvariableopO
Ksavev2_adam_graph_attention_sparse_attn_kernel_self_7_v_read_readvariableopP
Lsavev2_adam_graph_attention_sparse_attn_kernel_neigh_7_v_read_readvariableopG
Csavev2_adam_graph_attention_sparse_1_kernel_0_v_read_readvariableopE
Asavev2_adam_graph_attention_sparse_1_bias_0_v_read_readvariableopQ
Msavev2_adam_graph_attention_sparse_1_attn_kernel_self_0_v_read_readvariableopR
Nsavev2_adam_graph_attention_sparse_1_attn_kernel_neigh_0_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?K
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?J
value?JB?JzB8layer_with_weights-0/ig_delta/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/ig_non_exist_edge/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_0/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_0/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_0/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_0/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_1/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_1/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_1/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_2/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_2/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_2/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_3/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_3/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_3/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_3/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_4/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_4/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_4/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_4/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_5/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_5/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_5/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_5/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_6/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_6/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_6/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_6/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/kernel_7/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/bias_7/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/attn_kernel_self_7/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-0/attn_kernel_neigh_7/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/ig_delta/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/ig_non_exist_edge/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/kernel_0/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/bias_0/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/attn_kernel_self_0/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/attn_kernel_neigh_0/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-1/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh_7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/attn_kernel_self_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-1/attn_kernel_neigh_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?
value?B?zB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?A
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_graph_attention_sparse_ig_delta_read_readvariableopCsavev2_graph_attention_sparse_ig_non_exist_edge_read_readvariableop:savev2_graph_attention_sparse_kernel_0_read_readvariableop8savev2_graph_attention_sparse_bias_0_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_0_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_0_read_readvariableop:savev2_graph_attention_sparse_kernel_1_read_readvariableop8savev2_graph_attention_sparse_bias_1_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_1_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_1_read_readvariableop:savev2_graph_attention_sparse_kernel_2_read_readvariableop8savev2_graph_attention_sparse_bias_2_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_2_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_2_read_readvariableop:savev2_graph_attention_sparse_kernel_3_read_readvariableop8savev2_graph_attention_sparse_bias_3_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_3_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_3_read_readvariableop:savev2_graph_attention_sparse_kernel_4_read_readvariableop8savev2_graph_attention_sparse_bias_4_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_4_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_4_read_readvariableop:savev2_graph_attention_sparse_kernel_5_read_readvariableop8savev2_graph_attention_sparse_bias_5_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_5_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_5_read_readvariableop:savev2_graph_attention_sparse_kernel_6_read_readvariableop8savev2_graph_attention_sparse_bias_6_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_6_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_6_read_readvariableop:savev2_graph_attention_sparse_kernel_7_read_readvariableop8savev2_graph_attention_sparse_bias_7_read_readvariableopDsavev2_graph_attention_sparse_attn_kernel_self_7_read_readvariableopEsavev2_graph_attention_sparse_attn_kernel_neigh_7_read_readvariableop<savev2_graph_attention_sparse_1_ig_delta_read_readvariableopEsavev2_graph_attention_sparse_1_ig_non_exist_edge_read_readvariableop<savev2_graph_attention_sparse_1_kernel_0_read_readvariableop:savev2_graph_attention_sparse_1_bias_0_read_readvariableopFsavev2_graph_attention_sparse_1_attn_kernel_self_0_read_readvariableopGsavev2_graph_attention_sparse_1_attn_kernel_neigh_0_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_0_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_0_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_0_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_0_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_1_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_1_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_1_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_1_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_2_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_2_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_2_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_2_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_3_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_3_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_3_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_3_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_4_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_4_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_4_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_4_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_5_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_5_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_5_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_5_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_6_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_6_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_6_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_6_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_7_m_read_readvariableop?savev2_adam_graph_attention_sparse_bias_7_m_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_7_m_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_7_m_read_readvariableopCsavev2_adam_graph_attention_sparse_1_kernel_0_m_read_readvariableopAsavev2_adam_graph_attention_sparse_1_bias_0_m_read_readvariableopMsavev2_adam_graph_attention_sparse_1_attn_kernel_self_0_m_read_readvariableopNsavev2_adam_graph_attention_sparse_1_attn_kernel_neigh_0_m_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_0_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_0_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_0_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_0_v_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_1_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_1_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_1_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_1_v_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_2_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_2_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_2_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_2_v_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_3_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_3_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_3_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_3_v_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_4_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_4_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_4_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_4_v_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_5_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_5_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_5_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_5_v_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_6_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_6_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_6_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_6_v_read_readvariableopAsavev2_adam_graph_attention_sparse_kernel_7_v_read_readvariableop?savev2_adam_graph_attention_sparse_bias_7_v_read_readvariableopKsavev2_adam_graph_attention_sparse_attn_kernel_self_7_v_read_readvariableopLsavev2_adam_graph_attention_sparse_attn_kernel_neigh_7_v_read_readvariableopCsavev2_adam_graph_attention_sparse_1_kernel_0_v_read_readvariableopAsavev2_adam_graph_attention_sparse_1_bias_0_v_read_readvariableopMsavev2_adam_graph_attention_sparse_1_attn_kernel_self_0_v_read_readvariableopNsavev2_adam_graph_attention_sparse_1_attn_kernel_neigh_0_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes~
|2z	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :	?::::	?::::	?::::	?::::	?::::	?::::	?::::	?:::: : :@:::: : : : : : : : : :	?::::	?::::	?::::	?::::	?::::	?::::	?::::	?::::@::::	?::::	?::::	?::::	?::::	?::::	?::::	?::::	?::::@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?: 

_output_shapes
::$	 

_output_shapes

::$
 

_output_shapes

::%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?:  

_output_shapes
::$! 

_output_shapes

::$" 

_output_shapes

::#

_output_shapes
: :$

_output_shapes
: :$% 

_output_shapes

:@: &

_output_shapes
::$' 

_output_shapes

::$( 

_output_shapes

::)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :%2!

_output_shapes
:	?: 3

_output_shapes
::$4 

_output_shapes

::$5 

_output_shapes

::%6!

_output_shapes
:	?: 7

_output_shapes
::$8 

_output_shapes

::$9 

_output_shapes

::%:!

_output_shapes
:	?: ;

_output_shapes
::$< 

_output_shapes

::$= 

_output_shapes

::%>!

_output_shapes
:	?: ?

_output_shapes
::$@ 

_output_shapes

::$A 

_output_shapes

::%B!

_output_shapes
:	?: C

_output_shapes
::$D 

_output_shapes

::$E 

_output_shapes

::%F!

_output_shapes
:	?: G

_output_shapes
::$H 

_output_shapes

::$I 

_output_shapes

::%J!

_output_shapes
:	?: K

_output_shapes
::$L 

_output_shapes

::$M 

_output_shapes

::%N!

_output_shapes
:	?: O

_output_shapes
::$P 

_output_shapes

::$Q 

_output_shapes

::$R 

_output_shapes

:@: S

_output_shapes
::$T 

_output_shapes

::$U 

_output_shapes

::%V!

_output_shapes
:	?: W

_output_shapes
::$X 

_output_shapes

::$Y 

_output_shapes

::%Z!

_output_shapes
:	?: [

_output_shapes
::$\ 

_output_shapes

::$] 

_output_shapes

::%^!

_output_shapes
:	?: _

_output_shapes
::$` 

_output_shapes

::$a 

_output_shapes

::%b!

_output_shapes
:	?: c

_output_shapes
::$d 

_output_shapes

::$e 

_output_shapes

::%f!

_output_shapes
:	?: g

_output_shapes
::$h 

_output_shapes

::$i 

_output_shapes

::%j!

_output_shapes
:	?: k

_output_shapes
::$l 

_output_shapes

::$m 

_output_shapes

::%n!

_output_shapes
:	?: o

_output_shapes
::$p 

_output_shapes

::$q 

_output_shapes

::%r!

_output_shapes
:	?: s

_output_shapes
::$t 

_output_shapes

::$u 

_output_shapes

::$v 

_output_shapes

:@: w

_output_shapes
::$x 

_output_shapes

::$y 

_output_shapes

::z

_output_shapes
: 
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_9408

inputs

identity_1V
IdentityIdentityinputs*
T0*#
_output_shapes
:?@2

Identitye

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
:?@2

Identity_1"!

identity_1Identity_1:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
??
? 
@__inference_model_layer_call_and_return_conditional_losses_10611
inputs_0
inputs_1
inputs_2	
inputs_39
5graph_attention_sparse_matmul_readvariableop_resource;
7graph_attention_sparse_matmul_1_readvariableop_resource;
7graph_attention_sparse_matmul_2_readvariableop_resource:
6graph_attention_sparse_biasadd_readvariableop_resource;
7graph_attention_sparse_matmul_3_readvariableop_resource;
7graph_attention_sparse_matmul_4_readvariableop_resource;
7graph_attention_sparse_matmul_5_readvariableop_resource<
8graph_attention_sparse_biasadd_1_readvariableop_resource;
7graph_attention_sparse_matmul_6_readvariableop_resource;
7graph_attention_sparse_matmul_7_readvariableop_resource;
7graph_attention_sparse_matmul_8_readvariableop_resource<
8graph_attention_sparse_biasadd_2_readvariableop_resource;
7graph_attention_sparse_matmul_9_readvariableop_resource<
8graph_attention_sparse_matmul_10_readvariableop_resource<
8graph_attention_sparse_matmul_11_readvariableop_resource<
8graph_attention_sparse_biasadd_3_readvariableop_resource<
8graph_attention_sparse_matmul_12_readvariableop_resource<
8graph_attention_sparse_matmul_13_readvariableop_resource<
8graph_attention_sparse_matmul_14_readvariableop_resource<
8graph_attention_sparse_biasadd_4_readvariableop_resource<
8graph_attention_sparse_matmul_15_readvariableop_resource<
8graph_attention_sparse_matmul_16_readvariableop_resource<
8graph_attention_sparse_matmul_17_readvariableop_resource<
8graph_attention_sparse_biasadd_5_readvariableop_resource<
8graph_attention_sparse_matmul_18_readvariableop_resource<
8graph_attention_sparse_matmul_19_readvariableop_resource<
8graph_attention_sparse_matmul_20_readvariableop_resource<
8graph_attention_sparse_biasadd_6_readvariableop_resource<
8graph_attention_sparse_matmul_21_readvariableop_resource<
8graph_attention_sparse_matmul_22_readvariableop_resource<
8graph_attention_sparse_matmul_23_readvariableop_resource<
8graph_attention_sparse_biasadd_7_readvariableop_resource;
7graph_attention_sparse_1_matmul_readvariableop_resource=
9graph_attention_sparse_1_matmul_1_readvariableop_resource=
9graph_attention_sparse_1_matmul_2_readvariableop_resource<
8graph_attention_sparse_1_biasadd_readvariableop_resource
identity??-graph_attention_sparse/BiasAdd/ReadVariableOp?/graph_attention_sparse/BiasAdd_1/ReadVariableOp?/graph_attention_sparse/BiasAdd_2/ReadVariableOp?/graph_attention_sparse/BiasAdd_3/ReadVariableOp?/graph_attention_sparse/BiasAdd_4/ReadVariableOp?/graph_attention_sparse/BiasAdd_5/ReadVariableOp?/graph_attention_sparse/BiasAdd_6/ReadVariableOp?/graph_attention_sparse/BiasAdd_7/ReadVariableOp?,graph_attention_sparse/MatMul/ReadVariableOp?.graph_attention_sparse/MatMul_1/ReadVariableOp?/graph_attention_sparse/MatMul_10/ReadVariableOp?/graph_attention_sparse/MatMul_11/ReadVariableOp?/graph_attention_sparse/MatMul_12/ReadVariableOp?/graph_attention_sparse/MatMul_13/ReadVariableOp?/graph_attention_sparse/MatMul_14/ReadVariableOp?/graph_attention_sparse/MatMul_15/ReadVariableOp?/graph_attention_sparse/MatMul_16/ReadVariableOp?/graph_attention_sparse/MatMul_17/ReadVariableOp?/graph_attention_sparse/MatMul_18/ReadVariableOp?/graph_attention_sparse/MatMul_19/ReadVariableOp?.graph_attention_sparse/MatMul_2/ReadVariableOp?/graph_attention_sparse/MatMul_20/ReadVariableOp?/graph_attention_sparse/MatMul_21/ReadVariableOp?/graph_attention_sparse/MatMul_22/ReadVariableOp?/graph_attention_sparse/MatMul_23/ReadVariableOp?.graph_attention_sparse/MatMul_3/ReadVariableOp?.graph_attention_sparse/MatMul_4/ReadVariableOp?.graph_attention_sparse/MatMul_5/ReadVariableOp?.graph_attention_sparse/MatMul_6/ReadVariableOp?.graph_attention_sparse/MatMul_7/ReadVariableOp?.graph_attention_sparse/MatMul_8/ReadVariableOp?.graph_attention_sparse/MatMul_9/ReadVariableOp?/graph_attention_sparse_1/BiasAdd/ReadVariableOp?.graph_attention_sparse_1/MatMul/ReadVariableOp?0graph_attention_sparse_1/MatMul_1/ReadVariableOp?0graph_attention_sparse_1/MatMul_2/ReadVariableOp?
"squeezed_sparse_conversion/SqueezeSqueezeinputs_2*
T0	*'
_output_shapes
:?????????*
squeeze_dims
 2$
"squeezed_sparse_conversion/Squeeze?
$squeezed_sparse_conversion/Squeeze_1Squeezeinputs_3*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2&
$squeezed_sparse_conversion/Squeeze_1?
3squeezed_sparse_conversion/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      25
3squeezed_sparse_conversion/SparseTensor/dense_shapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulinputs_0dropout/dropout/Const:output:0*
T0*$
_output_shapes
:??2
dropout/dropout/Mul?
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?
  ?  2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*$
_output_shapes
:??*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:??2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:??2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*$
_output_shapes
:??2
dropout/dropout/Mul_1?
graph_attention_sparse/SqueezeSqueezedropout/dropout/Mul_1:z:0*
T0* 
_output_shapes
:
??*
squeeze_dims
 2 
graph_attention_sparse/Squeeze?
,graph_attention_sparse/MatMul/ReadVariableOpReadVariableOp5graph_attention_sparse_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,graph_attention_sparse/MatMul/ReadVariableOp?
graph_attention_sparse/MatMulMatMul'graph_attention_sparse/Squeeze:output:04graph_attention_sparse/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
graph_attention_sparse/MatMul?
.graph_attention_sparse/MatMul_1/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_1/ReadVariableOp?
graph_attention_sparse/MatMul_1MatMul'graph_attention_sparse/MatMul:product:06graph_attention_sparse/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_1?
.graph_attention_sparse/MatMul_2/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_2/ReadVariableOp?
graph_attention_sparse/MatMul_2MatMul'graph_attention_sparse/MatMul:product:06graph_attention_sparse/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_2?
$graph_attention_sparse/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$graph_attention_sparse/Reshape/shape?
graph_attention_sparse/ReshapeReshape)graph_attention_sparse/MatMul_1:product:0-graph_attention_sparse/Reshape/shape:output:0*
T0*
_output_shapes	
:?2 
graph_attention_sparse/Reshape?
*graph_attention_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*graph_attention_sparse/strided_slice/stack?
,graph_attention_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice/stack_1?
,graph_attention_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,graph_attention_sparse/strided_slice/stack_2?
$graph_attention_sparse/strided_sliceStridedSlice+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/strided_slice/stack:output:05graph_attention_sparse/strided_slice/stack_1:output:05graph_attention_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2&
$graph_attention_sparse/strided_slice?
$graph_attention_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$graph_attention_sparse/GatherV2/axis?
graph_attention_sparse/GatherV2GatherV2'graph_attention_sparse/Reshape:output:0-graph_attention_sparse/strided_slice:output:0-graph_attention_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2!
graph_attention_sparse/GatherV2?
&graph_attention_sparse/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_1/shape?
 graph_attention_sparse/Reshape_1Reshape)graph_attention_sparse/MatMul_2:product:0/graph_attention_sparse/Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_1?
,graph_attention_sparse/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_1/stack?
.graph_attention_sparse/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_1/stack_1?
.graph_attention_sparse/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_1/stack_2?
&graph_attention_sparse/strided_slice_1StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_1/stack:output:07graph_attention_sparse/strided_slice_1/stack_1:output:07graph_attention_sparse/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_1?
&graph_attention_sparse/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_1/axis?
!graph_attention_sparse/GatherV2_1GatherV2)graph_attention_sparse/Reshape_1:output:0/graph_attention_sparse/strided_slice_1:output:0/graph_attention_sparse/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_1?
graph_attention_sparse/addAddV2(graph_attention_sparse/GatherV2:output:0*graph_attention_sparse/GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add?
,graph_attention_sparse/leaky_re_lu/LeakyRelu	LeakyRelugraph_attention_sparse/add:z:0*#
_output_shapes
:?????????2.
,graph_attention_sparse/leaky_re_lu/LeakyRelu?
,graph_attention_sparse/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,graph_attention_sparse/dropout/dropout/Const?
*graph_attention_sparse/dropout/dropout/MulMul'graph_attention_sparse/MatMul:product:05graph_attention_sparse/dropout/dropout/Const:output:0*
T0*
_output_shapes
:	?2,
*graph_attention_sparse/dropout/dropout/Mul?
,graph_attention_sparse/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2.
,graph_attention_sparse/dropout/dropout/Shape?
Cgraph_attention_sparse/dropout/dropout/random_uniform/RandomUniformRandomUniform5graph_attention_sparse/dropout/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02E
Cgraph_attention_sparse/dropout/dropout/random_uniform/RandomUniform?
5graph_attention_sparse/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?27
5graph_attention_sparse/dropout/dropout/GreaterEqual/y?
3graph_attention_sparse/dropout/dropout/GreaterEqualGreaterEqualLgraph_attention_sparse/dropout/dropout/random_uniform/RandomUniform:output:0>graph_attention_sparse/dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?25
3graph_attention_sparse/dropout/dropout/GreaterEqual?
+graph_attention_sparse/dropout/dropout/CastCast7graph_attention_sparse/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2-
+graph_attention_sparse/dropout/dropout/Cast?
,graph_attention_sparse/dropout/dropout/Mul_1Mul.graph_attention_sparse/dropout/dropout/Mul:z:0/graph_attention_sparse/dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	?2.
,graph_attention_sparse/dropout/dropout/Mul_1?
.graph_attention_sparse/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_1/dropout/Const?
,graph_attention_sparse/dropout_1/dropout/MulMul:graph_attention_sparse/leaky_re_lu/LeakyRelu:activations:07graph_attention_sparse/dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2.
,graph_attention_sparse/dropout_1/dropout/Mul?
.graph_attention_sparse/dropout_1/dropout/ShapeShape:graph_attention_sparse/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:20
.graph_attention_sparse/dropout_1/dropout/Shape?
Egraph_attention_sparse/dropout_1/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02G
Egraph_attention_sparse/dropout_1/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_1/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_1/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_1/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????27
5graph_attention_sparse/dropout_1/dropout/GreaterEqual?
-graph_attention_sparse/dropout_1/dropout/CastCast9graph_attention_sparse/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_1/dropout/Cast?
.graph_attention_sparse/dropout_1/dropout/Mul_1Mul0graph_attention_sparse/dropout_1/dropout/Mul:z:01graph_attention_sparse/dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_1/dropout/Mul_1?
/graph_attention_sparse/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      21
/graph_attention_sparse/SparseTensor/dense_shape?
2graph_attention_sparse/SparseSoftmax/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_1/dropout/Mul_1:z:08graph_attention_sparse/SparseTensor/dense_shape:output:0*
T0*#
_output_shapes
:?????????24
2graph_attention_sparse/SparseSoftmax/SparseSoftmax?
Fgraph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0;graph_attention_sparse/SparseSoftmax/SparseSoftmax:output:08graph_attention_sparse/SparseTensor/dense_shape:output:00graph_attention_sparse/dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2H
Fgraph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
-graph_attention_sparse/BiasAdd/ReadVariableOpReadVariableOp6graph_attention_sparse_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-graph_attention_sparse/BiasAdd/ReadVariableOp?
graph_attention_sparse/BiasAddBiasAddPgraph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:05graph_attention_sparse/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2 
graph_attention_sparse/BiasAdd?
.graph_attention_sparse/MatMul_3/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_3_readvariableop_resource*
_output_shapes
:	?*
dtype020
.graph_attention_sparse/MatMul_3/ReadVariableOp?
graph_attention_sparse/MatMul_3MatMul'graph_attention_sparse/Squeeze:output:06graph_attention_sparse/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_3?
.graph_attention_sparse/MatMul_4/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_4_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_4/ReadVariableOp?
graph_attention_sparse/MatMul_4MatMul)graph_attention_sparse/MatMul_3:product:06graph_attention_sparse/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_4?
.graph_attention_sparse/MatMul_5/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_5_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_5/ReadVariableOp?
graph_attention_sparse/MatMul_5MatMul)graph_attention_sparse/MatMul_3:product:06graph_attention_sparse/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_5?
&graph_attention_sparse/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_2/shape?
 graph_attention_sparse/Reshape_2Reshape)graph_attention_sparse/MatMul_4:product:0/graph_attention_sparse/Reshape_2/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_2?
,graph_attention_sparse/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_2/stack?
.graph_attention_sparse/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_2/stack_1?
.graph_attention_sparse/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_2/stack_2?
&graph_attention_sparse/strided_slice_2StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_2/stack:output:07graph_attention_sparse/strided_slice_2/stack_1:output:07graph_attention_sparse/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_2?
&graph_attention_sparse/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_2/axis?
!graph_attention_sparse/GatherV2_2GatherV2)graph_attention_sparse/Reshape_2:output:0/graph_attention_sparse/strided_slice_2:output:0/graph_attention_sparse/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_2?
&graph_attention_sparse/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_3/shape?
 graph_attention_sparse/Reshape_3Reshape)graph_attention_sparse/MatMul_5:product:0/graph_attention_sparse/Reshape_3/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_3?
,graph_attention_sparse/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_3/stack?
.graph_attention_sparse/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_3/stack_1?
.graph_attention_sparse/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_3/stack_2?
&graph_attention_sparse/strided_slice_3StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_3/stack:output:07graph_attention_sparse/strided_slice_3/stack_1:output:07graph_attention_sparse/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_3?
&graph_attention_sparse/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_3/axis?
!graph_attention_sparse/GatherV2_3GatherV2)graph_attention_sparse/Reshape_3:output:0/graph_attention_sparse/strided_slice_3:output:0/graph_attention_sparse/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_3?
graph_attention_sparse/add_1AddV2*graph_attention_sparse/GatherV2_2:output:0*graph_attention_sparse/GatherV2_3:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_1?
.graph_attention_sparse/leaky_re_lu_1/LeakyRelu	LeakyRelu graph_attention_sparse/add_1:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_1/LeakyRelu?
.graph_attention_sparse/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_2/dropout/Const?
,graph_attention_sparse/dropout_2/dropout/MulMul)graph_attention_sparse/MatMul_3:product:07graph_attention_sparse/dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	?2.
,graph_attention_sparse/dropout_2/dropout/Mul?
.graph_attention_sparse/dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     20
.graph_attention_sparse/dropout_2/dropout/Shape?
Egraph_attention_sparse/dropout_2/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02G
Egraph_attention_sparse/dropout_2/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_2/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_2/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_2/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?27
5graph_attention_sparse/dropout_2/dropout/GreaterEqual?
-graph_attention_sparse/dropout_2/dropout/CastCast9graph_attention_sparse/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2/
-graph_attention_sparse/dropout_2/dropout/Cast?
.graph_attention_sparse/dropout_2/dropout/Mul_1Mul0graph_attention_sparse/dropout_2/dropout/Mul:z:01graph_attention_sparse/dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	?20
.graph_attention_sparse/dropout_2/dropout/Mul_1?
.graph_attention_sparse/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_3/dropout/Const?
,graph_attention_sparse/dropout_3/dropout/MulMul<graph_attention_sparse/leaky_re_lu_1/LeakyRelu:activations:07graph_attention_sparse/dropout_3/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2.
,graph_attention_sparse/dropout_3/dropout/Mul?
.graph_attention_sparse/dropout_3/dropout/ShapeShape<graph_attention_sparse/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:20
.graph_attention_sparse/dropout_3/dropout/Shape?
Egraph_attention_sparse/dropout_3/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_3/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02G
Egraph_attention_sparse/dropout_3/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_3/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_3/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_3/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_3/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????27
5graph_attention_sparse/dropout_3/dropout/GreaterEqual?
-graph_attention_sparse/dropout_3/dropout/CastCast9graph_attention_sparse/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_3/dropout/Cast?
.graph_attention_sparse/dropout_3/dropout/Mul_1Mul0graph_attention_sparse/dropout_3/dropout/Mul:z:01graph_attention_sparse/dropout_3/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_3/dropout/Mul_1?
1graph_attention_sparse/SparseTensor_1/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_1/dense_shape?
4graph_attention_sparse/SparseSoftmax_1/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_3/dropout/Mul_1:z:0:graph_attention_sparse/SparseTensor_1/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_1/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_1/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_1/dense_shape:output:02graph_attention_sparse/dropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_1/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_1_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_1/ReadVariableOp?
 graph_attention_sparse/BiasAdd_1BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_1?
.graph_attention_sparse/MatMul_6/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_6_readvariableop_resource*
_output_shapes
:	?*
dtype020
.graph_attention_sparse/MatMul_6/ReadVariableOp?
graph_attention_sparse/MatMul_6MatMul'graph_attention_sparse/Squeeze:output:06graph_attention_sparse/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_6?
.graph_attention_sparse/MatMul_7/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_7_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_7/ReadVariableOp?
graph_attention_sparse/MatMul_7MatMul)graph_attention_sparse/MatMul_6:product:06graph_attention_sparse/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_7?
.graph_attention_sparse/MatMul_8/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_8_readvariableop_resource*
_output_shapes

:*
dtype020
.graph_attention_sparse/MatMul_8/ReadVariableOp?
graph_attention_sparse/MatMul_8MatMul)graph_attention_sparse/MatMul_6:product:06graph_attention_sparse/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_8?
&graph_attention_sparse/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_4/shape?
 graph_attention_sparse/Reshape_4Reshape)graph_attention_sparse/MatMul_7:product:0/graph_attention_sparse/Reshape_4/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_4?
,graph_attention_sparse/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_4/stack?
.graph_attention_sparse/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_4/stack_1?
.graph_attention_sparse/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_4/stack_2?
&graph_attention_sparse/strided_slice_4StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_4/stack:output:07graph_attention_sparse/strided_slice_4/stack_1:output:07graph_attention_sparse/strided_slice_4/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_4?
&graph_attention_sparse/GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_4/axis?
!graph_attention_sparse/GatherV2_4GatherV2)graph_attention_sparse/Reshape_4:output:0/graph_attention_sparse/strided_slice_4:output:0/graph_attention_sparse/GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_4?
&graph_attention_sparse/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_5/shape?
 graph_attention_sparse/Reshape_5Reshape)graph_attention_sparse/MatMul_8:product:0/graph_attention_sparse/Reshape_5/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_5?
,graph_attention_sparse/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_5/stack?
.graph_attention_sparse/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_5/stack_1?
.graph_attention_sparse/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_5/stack_2?
&graph_attention_sparse/strided_slice_5StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_5/stack:output:07graph_attention_sparse/strided_slice_5/stack_1:output:07graph_attention_sparse/strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_5?
&graph_attention_sparse/GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_5/axis?
!graph_attention_sparse/GatherV2_5GatherV2)graph_attention_sparse/Reshape_5:output:0/graph_attention_sparse/strided_slice_5:output:0/graph_attention_sparse/GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_5?
graph_attention_sparse/add_2AddV2*graph_attention_sparse/GatherV2_4:output:0*graph_attention_sparse/GatherV2_5:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_2?
.graph_attention_sparse/leaky_re_lu_2/LeakyRelu	LeakyRelu graph_attention_sparse/add_2:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_2/LeakyRelu?
.graph_attention_sparse/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_4/dropout/Const?
,graph_attention_sparse/dropout_4/dropout/MulMul)graph_attention_sparse/MatMul_6:product:07graph_attention_sparse/dropout_4/dropout/Const:output:0*
T0*
_output_shapes
:	?2.
,graph_attention_sparse/dropout_4/dropout/Mul?
.graph_attention_sparse/dropout_4/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     20
.graph_attention_sparse/dropout_4/dropout/Shape?
Egraph_attention_sparse/dropout_4/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_4/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02G
Egraph_attention_sparse/dropout_4/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_4/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_4/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_4/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_4/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?27
5graph_attention_sparse/dropout_4/dropout/GreaterEqual?
-graph_attention_sparse/dropout_4/dropout/CastCast9graph_attention_sparse/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2/
-graph_attention_sparse/dropout_4/dropout/Cast?
.graph_attention_sparse/dropout_4/dropout/Mul_1Mul0graph_attention_sparse/dropout_4/dropout/Mul:z:01graph_attention_sparse/dropout_4/dropout/Cast:y:0*
T0*
_output_shapes
:	?20
.graph_attention_sparse/dropout_4/dropout/Mul_1?
.graph_attention_sparse/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_5/dropout/Const?
,graph_attention_sparse/dropout_5/dropout/MulMul<graph_attention_sparse/leaky_re_lu_2/LeakyRelu:activations:07graph_attention_sparse/dropout_5/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2.
,graph_attention_sparse/dropout_5/dropout/Mul?
.graph_attention_sparse/dropout_5/dropout/ShapeShape<graph_attention_sparse/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:20
.graph_attention_sparse/dropout_5/dropout/Shape?
Egraph_attention_sparse/dropout_5/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_5/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02G
Egraph_attention_sparse/dropout_5/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_5/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_5/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_5/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_5/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????27
5graph_attention_sparse/dropout_5/dropout/GreaterEqual?
-graph_attention_sparse/dropout_5/dropout/CastCast9graph_attention_sparse/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_5/dropout/Cast?
.graph_attention_sparse/dropout_5/dropout/Mul_1Mul0graph_attention_sparse/dropout_5/dropout/Mul:z:01graph_attention_sparse/dropout_5/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_5/dropout/Mul_1?
1graph_attention_sparse/SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_2/dense_shape?
4graph_attention_sparse/SparseSoftmax_2/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_5/dropout/Mul_1:z:0:graph_attention_sparse/SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_2/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_2/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_2/dense_shape:output:02graph_attention_sparse/dropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_2/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_2_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_2/ReadVariableOp?
 graph_attention_sparse/BiasAdd_2BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_2?
.graph_attention_sparse/MatMul_9/ReadVariableOpReadVariableOp7graph_attention_sparse_matmul_9_readvariableop_resource*
_output_shapes
:	?*
dtype020
.graph_attention_sparse/MatMul_9/ReadVariableOp?
graph_attention_sparse/MatMul_9MatMul'graph_attention_sparse/Squeeze:output:06graph_attention_sparse/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse/MatMul_9?
/graph_attention_sparse/MatMul_10/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_10_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_10/ReadVariableOp?
 graph_attention_sparse/MatMul_10MatMul)graph_attention_sparse/MatMul_9:product:07graph_attention_sparse/MatMul_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_10?
/graph_attention_sparse/MatMul_11/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_11_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_11/ReadVariableOp?
 graph_attention_sparse/MatMul_11MatMul)graph_attention_sparse/MatMul_9:product:07graph_attention_sparse/MatMul_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_11?
&graph_attention_sparse/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_6/shape?
 graph_attention_sparse/Reshape_6Reshape*graph_attention_sparse/MatMul_10:product:0/graph_attention_sparse/Reshape_6/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_6?
,graph_attention_sparse/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_6/stack?
.graph_attention_sparse/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_6/stack_1?
.graph_attention_sparse/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_6/stack_2?
&graph_attention_sparse/strided_slice_6StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_6/stack:output:07graph_attention_sparse/strided_slice_6/stack_1:output:07graph_attention_sparse/strided_slice_6/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_6?
&graph_attention_sparse/GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_6/axis?
!graph_attention_sparse/GatherV2_6GatherV2)graph_attention_sparse/Reshape_6:output:0/graph_attention_sparse/strided_slice_6:output:0/graph_attention_sparse/GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_6?
&graph_attention_sparse/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_7/shape?
 graph_attention_sparse/Reshape_7Reshape*graph_attention_sparse/MatMul_11:product:0/graph_attention_sparse/Reshape_7/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_7?
,graph_attention_sparse/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_7/stack?
.graph_attention_sparse/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_7/stack_1?
.graph_attention_sparse/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_7/stack_2?
&graph_attention_sparse/strided_slice_7StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_7/stack:output:07graph_attention_sparse/strided_slice_7/stack_1:output:07graph_attention_sparse/strided_slice_7/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_7?
&graph_attention_sparse/GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_7/axis?
!graph_attention_sparse/GatherV2_7GatherV2)graph_attention_sparse/Reshape_7:output:0/graph_attention_sparse/strided_slice_7:output:0/graph_attention_sparse/GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_7?
graph_attention_sparse/add_3AddV2*graph_attention_sparse/GatherV2_6:output:0*graph_attention_sparse/GatherV2_7:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_3?
.graph_attention_sparse/leaky_re_lu_3/LeakyRelu	LeakyRelu graph_attention_sparse/add_3:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_3/LeakyRelu?
.graph_attention_sparse/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_6/dropout/Const?
,graph_attention_sparse/dropout_6/dropout/MulMul)graph_attention_sparse/MatMul_9:product:07graph_attention_sparse/dropout_6/dropout/Const:output:0*
T0*
_output_shapes
:	?2.
,graph_attention_sparse/dropout_6/dropout/Mul?
.graph_attention_sparse/dropout_6/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     20
.graph_attention_sparse/dropout_6/dropout/Shape?
Egraph_attention_sparse/dropout_6/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_6/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02G
Egraph_attention_sparse/dropout_6/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_6/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_6/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_6/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_6/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?27
5graph_attention_sparse/dropout_6/dropout/GreaterEqual?
-graph_attention_sparse/dropout_6/dropout/CastCast9graph_attention_sparse/dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2/
-graph_attention_sparse/dropout_6/dropout/Cast?
.graph_attention_sparse/dropout_6/dropout/Mul_1Mul0graph_attention_sparse/dropout_6/dropout/Mul:z:01graph_attention_sparse/dropout_6/dropout/Cast:y:0*
T0*
_output_shapes
:	?20
.graph_attention_sparse/dropout_6/dropout/Mul_1?
.graph_attention_sparse/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_7/dropout/Const?
,graph_attention_sparse/dropout_7/dropout/MulMul<graph_attention_sparse/leaky_re_lu_3/LeakyRelu:activations:07graph_attention_sparse/dropout_7/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2.
,graph_attention_sparse/dropout_7/dropout/Mul?
.graph_attention_sparse/dropout_7/dropout/ShapeShape<graph_attention_sparse/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:20
.graph_attention_sparse/dropout_7/dropout/Shape?
Egraph_attention_sparse/dropout_7/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_7/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02G
Egraph_attention_sparse/dropout_7/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_7/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_7/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_7/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_7/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????27
5graph_attention_sparse/dropout_7/dropout/GreaterEqual?
-graph_attention_sparse/dropout_7/dropout/CastCast9graph_attention_sparse/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_7/dropout/Cast?
.graph_attention_sparse/dropout_7/dropout/Mul_1Mul0graph_attention_sparse/dropout_7/dropout/Mul:z:01graph_attention_sparse/dropout_7/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_7/dropout/Mul_1?
1graph_attention_sparse/SparseTensor_3/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_3/dense_shape?
4graph_attention_sparse/SparseSoftmax_3/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_7/dropout/Mul_1:z:0:graph_attention_sparse/SparseTensor_3/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_3/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_3/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_3/dense_shape:output:02graph_attention_sparse/dropout_6/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_3/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_3_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_3/ReadVariableOp?
 graph_attention_sparse/BiasAdd_3BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_3?
/graph_attention_sparse/MatMul_12/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_12_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_12/ReadVariableOp?
 graph_attention_sparse/MatMul_12MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_12?
/graph_attention_sparse/MatMul_13/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_13_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_13/ReadVariableOp?
 graph_attention_sparse/MatMul_13MatMul*graph_attention_sparse/MatMul_12:product:07graph_attention_sparse/MatMul_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_13?
/graph_attention_sparse/MatMul_14/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_14_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_14/ReadVariableOp?
 graph_attention_sparse/MatMul_14MatMul*graph_attention_sparse/MatMul_12:product:07graph_attention_sparse/MatMul_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_14?
&graph_attention_sparse/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_8/shape?
 graph_attention_sparse/Reshape_8Reshape*graph_attention_sparse/MatMul_13:product:0/graph_attention_sparse/Reshape_8/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_8?
,graph_attention_sparse/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse/strided_slice_8/stack?
.graph_attention_sparse/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_8/stack_1?
.graph_attention_sparse/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_8/stack_2?
&graph_attention_sparse/strided_slice_8StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_8/stack:output:07graph_attention_sparse/strided_slice_8/stack_1:output:07graph_attention_sparse/strided_slice_8/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_8?
&graph_attention_sparse/GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_8/axis?
!graph_attention_sparse/GatherV2_8GatherV2)graph_attention_sparse/Reshape_8:output:0/graph_attention_sparse/strided_slice_8:output:0/graph_attention_sparse/GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_8?
&graph_attention_sparse/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse/Reshape_9/shape?
 graph_attention_sparse/Reshape_9Reshape*graph_attention_sparse/MatMul_14:product:0/graph_attention_sparse/Reshape_9/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse/Reshape_9?
,graph_attention_sparse/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,graph_attention_sparse/strided_slice_9/stack?
.graph_attention_sparse/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse/strided_slice_9/stack_1?
.graph_attention_sparse/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse/strided_slice_9/stack_2?
&graph_attention_sparse/strided_slice_9StridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse/strided_slice_9/stack:output:07graph_attention_sparse/strided_slice_9/stack_1:output:07graph_attention_sparse/strided_slice_9/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse/strided_slice_9?
&graph_attention_sparse/GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse/GatherV2_9/axis?
!graph_attention_sparse/GatherV2_9GatherV2)graph_attention_sparse/Reshape_9:output:0/graph_attention_sparse/strided_slice_9:output:0/graph_attention_sparse/GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse/GatherV2_9?
graph_attention_sparse/add_4AddV2*graph_attention_sparse/GatherV2_8:output:0*graph_attention_sparse/GatherV2_9:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_4?
.graph_attention_sparse/leaky_re_lu_4/LeakyRelu	LeakyRelu graph_attention_sparse/add_4:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_4/LeakyRelu?
.graph_attention_sparse/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_8/dropout/Const?
,graph_attention_sparse/dropout_8/dropout/MulMul*graph_attention_sparse/MatMul_12:product:07graph_attention_sparse/dropout_8/dropout/Const:output:0*
T0*
_output_shapes
:	?2.
,graph_attention_sparse/dropout_8/dropout/Mul?
.graph_attention_sparse/dropout_8/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     20
.graph_attention_sparse/dropout_8/dropout/Shape?
Egraph_attention_sparse/dropout_8/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_8/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02G
Egraph_attention_sparse/dropout_8/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_8/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_8/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_8/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_8/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?27
5graph_attention_sparse/dropout_8/dropout/GreaterEqual?
-graph_attention_sparse/dropout_8/dropout/CastCast9graph_attention_sparse/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2/
-graph_attention_sparse/dropout_8/dropout/Cast?
.graph_attention_sparse/dropout_8/dropout/Mul_1Mul0graph_attention_sparse/dropout_8/dropout/Mul:z:01graph_attention_sparse/dropout_8/dropout/Cast:y:0*
T0*
_output_shapes
:	?20
.graph_attention_sparse/dropout_8/dropout/Mul_1?
.graph_attention_sparse/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.graph_attention_sparse/dropout_9/dropout/Const?
,graph_attention_sparse/dropout_9/dropout/MulMul<graph_attention_sparse/leaky_re_lu_4/LeakyRelu:activations:07graph_attention_sparse/dropout_9/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2.
,graph_attention_sparse/dropout_9/dropout/Mul?
.graph_attention_sparse/dropout_9/dropout/ShapeShape<graph_attention_sparse/leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:20
.graph_attention_sparse/dropout_9/dropout/Shape?
Egraph_attention_sparse/dropout_9/dropout/random_uniform/RandomUniformRandomUniform7graph_attention_sparse/dropout_9/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02G
Egraph_attention_sparse/dropout_9/dropout/random_uniform/RandomUniform?
7graph_attention_sparse/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?29
7graph_attention_sparse/dropout_9/dropout/GreaterEqual/y?
5graph_attention_sparse/dropout_9/dropout/GreaterEqualGreaterEqualNgraph_attention_sparse/dropout_9/dropout/random_uniform/RandomUniform:output:0@graph_attention_sparse/dropout_9/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????27
5graph_attention_sparse/dropout_9/dropout/GreaterEqual?
-graph_attention_sparse/dropout_9/dropout/CastCast9graph_attention_sparse/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_9/dropout/Cast?
.graph_attention_sparse/dropout_9/dropout/Mul_1Mul0graph_attention_sparse/dropout_9/dropout/Mul:z:01graph_attention_sparse/dropout_9/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_9/dropout/Mul_1?
1graph_attention_sparse/SparseTensor_4/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_4/dense_shape?
4graph_attention_sparse/SparseSoftmax_4/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:02graph_attention_sparse/dropout_9/dropout/Mul_1:z:0:graph_attention_sparse/SparseTensor_4/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_4/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_4/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_4/dense_shape:output:02graph_attention_sparse/dropout_8/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_4/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_4_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_4/ReadVariableOp?
 graph_attention_sparse/BiasAdd_4BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_4?
/graph_attention_sparse/MatMul_15/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_15_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_15/ReadVariableOp?
 graph_attention_sparse/MatMul_15MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_15?
/graph_attention_sparse/MatMul_16/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_16_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_16/ReadVariableOp?
 graph_attention_sparse/MatMul_16MatMul*graph_attention_sparse/MatMul_15:product:07graph_attention_sparse/MatMul_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_16?
/graph_attention_sparse/MatMul_17/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_17_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_17/ReadVariableOp?
 graph_attention_sparse/MatMul_17MatMul*graph_attention_sparse/MatMul_15:product:07graph_attention_sparse/MatMul_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_17?
'graph_attention_sparse/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_10/shape?
!graph_attention_sparse/Reshape_10Reshape*graph_attention_sparse/MatMul_16:product:00graph_attention_sparse/Reshape_10/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_10?
-graph_attention_sparse/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-graph_attention_sparse/strided_slice_10/stack?
/graph_attention_sparse/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_10/stack_1?
/graph_attention_sparse/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_10/stack_2?
'graph_attention_sparse/strided_slice_10StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_10/stack:output:08graph_attention_sparse/strided_slice_10/stack_1:output:08graph_attention_sparse/strided_slice_10/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_10?
'graph_attention_sparse/GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_10/axis?
"graph_attention_sparse/GatherV2_10GatherV2*graph_attention_sparse/Reshape_10:output:00graph_attention_sparse/strided_slice_10:output:00graph_attention_sparse/GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_10?
'graph_attention_sparse/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_11/shape?
!graph_attention_sparse/Reshape_11Reshape*graph_attention_sparse/MatMul_17:product:00graph_attention_sparse/Reshape_11/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_11?
-graph_attention_sparse/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-graph_attention_sparse/strided_slice_11/stack?
/graph_attention_sparse/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_11/stack_1?
/graph_attention_sparse/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_11/stack_2?
'graph_attention_sparse/strided_slice_11StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_11/stack:output:08graph_attention_sparse/strided_slice_11/stack_1:output:08graph_attention_sparse/strided_slice_11/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_11?
'graph_attention_sparse/GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_11/axis?
"graph_attention_sparse/GatherV2_11GatherV2*graph_attention_sparse/Reshape_11:output:00graph_attention_sparse/strided_slice_11:output:00graph_attention_sparse/GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_11?
graph_attention_sparse/add_5AddV2+graph_attention_sparse/GatherV2_10:output:0+graph_attention_sparse/GatherV2_11:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_5?
.graph_attention_sparse/leaky_re_lu_5/LeakyRelu	LeakyRelu graph_attention_sparse/add_5:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_5/LeakyRelu?
/graph_attention_sparse/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @21
/graph_attention_sparse/dropout_10/dropout/Const?
-graph_attention_sparse/dropout_10/dropout/MulMul*graph_attention_sparse/MatMul_15:product:08graph_attention_sparse/dropout_10/dropout/Const:output:0*
T0*
_output_shapes
:	?2/
-graph_attention_sparse/dropout_10/dropout/Mul?
/graph_attention_sparse/dropout_10/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     21
/graph_attention_sparse/dropout_10/dropout/Shape?
Fgraph_attention_sparse/dropout_10/dropout/random_uniform/RandomUniformRandomUniform8graph_attention_sparse/dropout_10/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02H
Fgraph_attention_sparse/dropout_10/dropout/random_uniform/RandomUniform?
8graph_attention_sparse/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2:
8graph_attention_sparse/dropout_10/dropout/GreaterEqual/y?
6graph_attention_sparse/dropout_10/dropout/GreaterEqualGreaterEqualOgraph_attention_sparse/dropout_10/dropout/random_uniform/RandomUniform:output:0Agraph_attention_sparse/dropout_10/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?28
6graph_attention_sparse/dropout_10/dropout/GreaterEqual?
.graph_attention_sparse/dropout_10/dropout/CastCast:graph_attention_sparse/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?20
.graph_attention_sparse/dropout_10/dropout/Cast?
/graph_attention_sparse/dropout_10/dropout/Mul_1Mul1graph_attention_sparse/dropout_10/dropout/Mul:z:02graph_attention_sparse/dropout_10/dropout/Cast:y:0*
T0*
_output_shapes
:	?21
/graph_attention_sparse/dropout_10/dropout/Mul_1?
/graph_attention_sparse/dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @21
/graph_attention_sparse/dropout_11/dropout/Const?
-graph_attention_sparse/dropout_11/dropout/MulMul<graph_attention_sparse/leaky_re_lu_5/LeakyRelu:activations:08graph_attention_sparse/dropout_11/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_11/dropout/Mul?
/graph_attention_sparse/dropout_11/dropout/ShapeShape<graph_attention_sparse/leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:21
/graph_attention_sparse/dropout_11/dropout/Shape?
Fgraph_attention_sparse/dropout_11/dropout/random_uniform/RandomUniformRandomUniform8graph_attention_sparse/dropout_11/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02H
Fgraph_attention_sparse/dropout_11/dropout/random_uniform/RandomUniform?
8graph_attention_sparse/dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2:
8graph_attention_sparse/dropout_11/dropout/GreaterEqual/y?
6graph_attention_sparse/dropout_11/dropout/GreaterEqualGreaterEqualOgraph_attention_sparse/dropout_11/dropout/random_uniform/RandomUniform:output:0Agraph_attention_sparse/dropout_11/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????28
6graph_attention_sparse/dropout_11/dropout/GreaterEqual?
.graph_attention_sparse/dropout_11/dropout/CastCast:graph_attention_sparse/dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_11/dropout/Cast?
/graph_attention_sparse/dropout_11/dropout/Mul_1Mul1graph_attention_sparse/dropout_11/dropout/Mul:z:02graph_attention_sparse/dropout_11/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????21
/graph_attention_sparse/dropout_11/dropout/Mul_1?
1graph_attention_sparse/SparseTensor_5/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_5/dense_shape?
4graph_attention_sparse/SparseSoftmax_5/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/dropout_11/dropout/Mul_1:z:0:graph_attention_sparse/SparseTensor_5/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_5/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_5/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_5/dense_shape:output:03graph_attention_sparse/dropout_10/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_5/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_5_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_5/ReadVariableOp?
 graph_attention_sparse/BiasAdd_5BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_5?
/graph_attention_sparse/MatMul_18/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_18_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_18/ReadVariableOp?
 graph_attention_sparse/MatMul_18MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_18/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_18?
/graph_attention_sparse/MatMul_19/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_19_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_19/ReadVariableOp?
 graph_attention_sparse/MatMul_19MatMul*graph_attention_sparse/MatMul_18:product:07graph_attention_sparse/MatMul_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_19?
/graph_attention_sparse/MatMul_20/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_20_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_20/ReadVariableOp?
 graph_attention_sparse/MatMul_20MatMul*graph_attention_sparse/MatMul_18:product:07graph_attention_sparse/MatMul_20/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_20?
'graph_attention_sparse/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_12/shape?
!graph_attention_sparse/Reshape_12Reshape*graph_attention_sparse/MatMul_19:product:00graph_attention_sparse/Reshape_12/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_12?
-graph_attention_sparse/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-graph_attention_sparse/strided_slice_12/stack?
/graph_attention_sparse/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_12/stack_1?
/graph_attention_sparse/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_12/stack_2?
'graph_attention_sparse/strided_slice_12StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_12/stack:output:08graph_attention_sparse/strided_slice_12/stack_1:output:08graph_attention_sparse/strided_slice_12/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_12?
'graph_attention_sparse/GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_12/axis?
"graph_attention_sparse/GatherV2_12GatherV2*graph_attention_sparse/Reshape_12:output:00graph_attention_sparse/strided_slice_12:output:00graph_attention_sparse/GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_12?
'graph_attention_sparse/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_13/shape?
!graph_attention_sparse/Reshape_13Reshape*graph_attention_sparse/MatMul_20:product:00graph_attention_sparse/Reshape_13/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_13?
-graph_attention_sparse/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-graph_attention_sparse/strided_slice_13/stack?
/graph_attention_sparse/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_13/stack_1?
/graph_attention_sparse/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_13/stack_2?
'graph_attention_sparse/strided_slice_13StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_13/stack:output:08graph_attention_sparse/strided_slice_13/stack_1:output:08graph_attention_sparse/strided_slice_13/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_13?
'graph_attention_sparse/GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_13/axis?
"graph_attention_sparse/GatherV2_13GatherV2*graph_attention_sparse/Reshape_13:output:00graph_attention_sparse/strided_slice_13:output:00graph_attention_sparse/GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_13?
graph_attention_sparse/add_6AddV2+graph_attention_sparse/GatherV2_12:output:0+graph_attention_sparse/GatherV2_13:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_6?
.graph_attention_sparse/leaky_re_lu_6/LeakyRelu	LeakyRelu graph_attention_sparse/add_6:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_6/LeakyRelu?
/graph_attention_sparse/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @21
/graph_attention_sparse/dropout_12/dropout/Const?
-graph_attention_sparse/dropout_12/dropout/MulMul*graph_attention_sparse/MatMul_18:product:08graph_attention_sparse/dropout_12/dropout/Const:output:0*
T0*
_output_shapes
:	?2/
-graph_attention_sparse/dropout_12/dropout/Mul?
/graph_attention_sparse/dropout_12/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     21
/graph_attention_sparse/dropout_12/dropout/Shape?
Fgraph_attention_sparse/dropout_12/dropout/random_uniform/RandomUniformRandomUniform8graph_attention_sparse/dropout_12/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02H
Fgraph_attention_sparse/dropout_12/dropout/random_uniform/RandomUniform?
8graph_attention_sparse/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2:
8graph_attention_sparse/dropout_12/dropout/GreaterEqual/y?
6graph_attention_sparse/dropout_12/dropout/GreaterEqualGreaterEqualOgraph_attention_sparse/dropout_12/dropout/random_uniform/RandomUniform:output:0Agraph_attention_sparse/dropout_12/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?28
6graph_attention_sparse/dropout_12/dropout/GreaterEqual?
.graph_attention_sparse/dropout_12/dropout/CastCast:graph_attention_sparse/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?20
.graph_attention_sparse/dropout_12/dropout/Cast?
/graph_attention_sparse/dropout_12/dropout/Mul_1Mul1graph_attention_sparse/dropout_12/dropout/Mul:z:02graph_attention_sparse/dropout_12/dropout/Cast:y:0*
T0*
_output_shapes
:	?21
/graph_attention_sparse/dropout_12/dropout/Mul_1?
/graph_attention_sparse/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @21
/graph_attention_sparse/dropout_13/dropout/Const?
-graph_attention_sparse/dropout_13/dropout/MulMul<graph_attention_sparse/leaky_re_lu_6/LeakyRelu:activations:08graph_attention_sparse/dropout_13/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_13/dropout/Mul?
/graph_attention_sparse/dropout_13/dropout/ShapeShape<graph_attention_sparse/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:21
/graph_attention_sparse/dropout_13/dropout/Shape?
Fgraph_attention_sparse/dropout_13/dropout/random_uniform/RandomUniformRandomUniform8graph_attention_sparse/dropout_13/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02H
Fgraph_attention_sparse/dropout_13/dropout/random_uniform/RandomUniform?
8graph_attention_sparse/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2:
8graph_attention_sparse/dropout_13/dropout/GreaterEqual/y?
6graph_attention_sparse/dropout_13/dropout/GreaterEqualGreaterEqualOgraph_attention_sparse/dropout_13/dropout/random_uniform/RandomUniform:output:0Agraph_attention_sparse/dropout_13/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????28
6graph_attention_sparse/dropout_13/dropout/GreaterEqual?
.graph_attention_sparse/dropout_13/dropout/CastCast:graph_attention_sparse/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_13/dropout/Cast?
/graph_attention_sparse/dropout_13/dropout/Mul_1Mul1graph_attention_sparse/dropout_13/dropout/Mul:z:02graph_attention_sparse/dropout_13/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????21
/graph_attention_sparse/dropout_13/dropout/Mul_1?
1graph_attention_sparse/SparseTensor_6/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_6/dense_shape?
4graph_attention_sparse/SparseSoftmax_6/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/dropout_13/dropout/Mul_1:z:0:graph_attention_sparse/SparseTensor_6/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_6/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_6/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_6/dense_shape:output:03graph_attention_sparse/dropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_6/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_6_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_6/ReadVariableOp?
 graph_attention_sparse/BiasAdd_6BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_6?
/graph_attention_sparse/MatMul_21/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_21_readvariableop_resource*
_output_shapes
:	?*
dtype021
/graph_attention_sparse/MatMul_21/ReadVariableOp?
 graph_attention_sparse/MatMul_21MatMul'graph_attention_sparse/Squeeze:output:07graph_attention_sparse/MatMul_21/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_21?
/graph_attention_sparse/MatMul_22/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_22_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_22/ReadVariableOp?
 graph_attention_sparse/MatMul_22MatMul*graph_attention_sparse/MatMul_21:product:07graph_attention_sparse/MatMul_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_22?
/graph_attention_sparse/MatMul_23/ReadVariableOpReadVariableOp8graph_attention_sparse_matmul_23_readvariableop_resource*
_output_shapes

:*
dtype021
/graph_attention_sparse/MatMul_23/ReadVariableOp?
 graph_attention_sparse/MatMul_23MatMul*graph_attention_sparse/MatMul_21:product:07graph_attention_sparse/MatMul_23/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/MatMul_23?
'graph_attention_sparse/Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_14/shape?
!graph_attention_sparse/Reshape_14Reshape*graph_attention_sparse/MatMul_22:product:00graph_attention_sparse/Reshape_14/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_14?
-graph_attention_sparse/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-graph_attention_sparse/strided_slice_14/stack?
/graph_attention_sparse/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_14/stack_1?
/graph_attention_sparse/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_14/stack_2?
'graph_attention_sparse/strided_slice_14StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_14/stack:output:08graph_attention_sparse/strided_slice_14/stack_1:output:08graph_attention_sparse/strided_slice_14/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_14?
'graph_attention_sparse/GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_14/axis?
"graph_attention_sparse/GatherV2_14GatherV2*graph_attention_sparse/Reshape_14:output:00graph_attention_sparse/strided_slice_14:output:00graph_attention_sparse/GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_14?
'graph_attention_sparse/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'graph_attention_sparse/Reshape_15/shape?
!graph_attention_sparse/Reshape_15Reshape*graph_attention_sparse/MatMul_23:product:00graph_attention_sparse/Reshape_15/shape:output:0*
T0*
_output_shapes	
:?2#
!graph_attention_sparse/Reshape_15?
-graph_attention_sparse/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-graph_attention_sparse/strided_slice_15/stack?
/graph_attention_sparse/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/graph_attention_sparse/strided_slice_15/stack_1?
/graph_attention_sparse/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/graph_attention_sparse/strided_slice_15/stack_2?
'graph_attention_sparse/strided_slice_15StridedSlice+squeezed_sparse_conversion/Squeeze:output:06graph_attention_sparse/strided_slice_15/stack:output:08graph_attention_sparse/strided_slice_15/stack_1:output:08graph_attention_sparse/strided_slice_15/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2)
'graph_attention_sparse/strided_slice_15?
'graph_attention_sparse/GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse/GatherV2_15/axis?
"graph_attention_sparse/GatherV2_15GatherV2*graph_attention_sparse/Reshape_15:output:00graph_attention_sparse/strided_slice_15:output:00graph_attention_sparse/GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2$
"graph_attention_sparse/GatherV2_15?
graph_attention_sparse/add_7AddV2+graph_attention_sparse/GatherV2_14:output:0+graph_attention_sparse/GatherV2_15:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse/add_7?
.graph_attention_sparse/leaky_re_lu_7/LeakyRelu	LeakyRelu graph_attention_sparse/add_7:z:0*#
_output_shapes
:?????????20
.graph_attention_sparse/leaky_re_lu_7/LeakyRelu?
/graph_attention_sparse/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @21
/graph_attention_sparse/dropout_14/dropout/Const?
-graph_attention_sparse/dropout_14/dropout/MulMul*graph_attention_sparse/MatMul_21:product:08graph_attention_sparse/dropout_14/dropout/Const:output:0*
T0*
_output_shapes
:	?2/
-graph_attention_sparse/dropout_14/dropout/Mul?
/graph_attention_sparse/dropout_14/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     21
/graph_attention_sparse/dropout_14/dropout/Shape?
Fgraph_attention_sparse/dropout_14/dropout/random_uniform/RandomUniformRandomUniform8graph_attention_sparse/dropout_14/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02H
Fgraph_attention_sparse/dropout_14/dropout/random_uniform/RandomUniform?
8graph_attention_sparse/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2:
8graph_attention_sparse/dropout_14/dropout/GreaterEqual/y?
6graph_attention_sparse/dropout_14/dropout/GreaterEqualGreaterEqualOgraph_attention_sparse/dropout_14/dropout/random_uniform/RandomUniform:output:0Agraph_attention_sparse/dropout_14/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?28
6graph_attention_sparse/dropout_14/dropout/GreaterEqual?
.graph_attention_sparse/dropout_14/dropout/CastCast:graph_attention_sparse/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?20
.graph_attention_sparse/dropout_14/dropout/Cast?
/graph_attention_sparse/dropout_14/dropout/Mul_1Mul1graph_attention_sparse/dropout_14/dropout/Mul:z:02graph_attention_sparse/dropout_14/dropout/Cast:y:0*
T0*
_output_shapes
:	?21
/graph_attention_sparse/dropout_14/dropout/Mul_1?
/graph_attention_sparse/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @21
/graph_attention_sparse/dropout_15/dropout/Const?
-graph_attention_sparse/dropout_15/dropout/MulMul<graph_attention_sparse/leaky_re_lu_7/LeakyRelu:activations:08graph_attention_sparse/dropout_15/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2/
-graph_attention_sparse/dropout_15/dropout/Mul?
/graph_attention_sparse/dropout_15/dropout/ShapeShape<graph_attention_sparse/leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:21
/graph_attention_sparse/dropout_15/dropout/Shape?
Fgraph_attention_sparse/dropout_15/dropout/random_uniform/RandomUniformRandomUniform8graph_attention_sparse/dropout_15/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02H
Fgraph_attention_sparse/dropout_15/dropout/random_uniform/RandomUniform?
8graph_attention_sparse/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2:
8graph_attention_sparse/dropout_15/dropout/GreaterEqual/y?
6graph_attention_sparse/dropout_15/dropout/GreaterEqualGreaterEqualOgraph_attention_sparse/dropout_15/dropout/random_uniform/RandomUniform:output:0Agraph_attention_sparse/dropout_15/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????28
6graph_attention_sparse/dropout_15/dropout/GreaterEqual?
.graph_attention_sparse/dropout_15/dropout/CastCast:graph_attention_sparse/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????20
.graph_attention_sparse/dropout_15/dropout/Cast?
/graph_attention_sparse/dropout_15/dropout/Mul_1Mul1graph_attention_sparse/dropout_15/dropout/Mul:z:02graph_attention_sparse/dropout_15/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????21
/graph_attention_sparse/dropout_15/dropout/Mul_1?
1graph_attention_sparse/SparseTensor_7/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse/SparseTensor_7/dense_shape?
4graph_attention_sparse/SparseSoftmax_7/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:03graph_attention_sparse/dropout_15/dropout/Mul_1:z:0:graph_attention_sparse/SparseTensor_7/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse/SparseSoftmax_7/SparseSoftmax?
Hgraph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse/SparseSoftmax_7/SparseSoftmax:output:0:graph_attention_sparse/SparseTensor_7/dense_shape:output:03graph_attention_sparse/dropout_14/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMul?
/graph_attention_sparse/BiasAdd_7/ReadVariableOpReadVariableOp8graph_attention_sparse_biasadd_7_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse/BiasAdd_7/ReadVariableOp?
 graph_attention_sparse/BiasAdd_7BiasAddRgraph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMul:product:07graph_attention_sparse/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse/BiasAdd_7?
"graph_attention_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"graph_attention_sparse/concat/axis?
graph_attention_sparse/concatConcatV2'graph_attention_sparse/BiasAdd:output:0)graph_attention_sparse/BiasAdd_1:output:0)graph_attention_sparse/BiasAdd_2:output:0)graph_attention_sparse/BiasAdd_3:output:0)graph_attention_sparse/BiasAdd_4:output:0)graph_attention_sparse/BiasAdd_5:output:0)graph_attention_sparse/BiasAdd_6:output:0)graph_attention_sparse/BiasAdd_7:output:0+graph_attention_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:	?@2
graph_attention_sparse/concat?
graph_attention_sparse/EluElu&graph_attention_sparse/concat:output:0*
T0*
_output_shapes
:	?@2
graph_attention_sparse/Elu?
%graph_attention_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%graph_attention_sparse/ExpandDims/dim?
!graph_attention_sparse/ExpandDims
ExpandDims(graph_attention_sparse/Elu:activations:0.graph_attention_sparse/ExpandDims/dim:output:0*
T0*#
_output_shapes
:?@2#
!graph_attention_sparse/ExpandDimsw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul*graph_attention_sparse/ExpandDims:output:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
:?@2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?
  @   2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
:?@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?@2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?@2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
:?@2
dropout_1/dropout/Mul_1?
 graph_attention_sparse_1/SqueezeSqueezedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?@*
squeeze_dims
 2"
 graph_attention_sparse_1/Squeeze?
.graph_attention_sparse_1/MatMul/ReadVariableOpReadVariableOp7graph_attention_sparse_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.graph_attention_sparse_1/MatMul/ReadVariableOp?
graph_attention_sparse_1/MatMulMatMul)graph_attention_sparse_1/Squeeze:output:06graph_attention_sparse_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
graph_attention_sparse_1/MatMul?
0graph_attention_sparse_1/MatMul_1/ReadVariableOpReadVariableOp9graph_attention_sparse_1_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype022
0graph_attention_sparse_1/MatMul_1/ReadVariableOp?
!graph_attention_sparse_1/MatMul_1MatMul)graph_attention_sparse_1/MatMul:product:08graph_attention_sparse_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!graph_attention_sparse_1/MatMul_1?
0graph_attention_sparse_1/MatMul_2/ReadVariableOpReadVariableOp9graph_attention_sparse_1_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype022
0graph_attention_sparse_1/MatMul_2/ReadVariableOp?
!graph_attention_sparse_1/MatMul_2MatMul)graph_attention_sparse_1/MatMul:product:08graph_attention_sparse_1/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!graph_attention_sparse_1/MatMul_2?
&graph_attention_sparse_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&graph_attention_sparse_1/Reshape/shape?
 graph_attention_sparse_1/ReshapeReshape+graph_attention_sparse_1/MatMul_1:product:0/graph_attention_sparse_1/Reshape/shape:output:0*
T0*
_output_shapes	
:?2"
 graph_attention_sparse_1/Reshape?
,graph_attention_sparse_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,graph_attention_sparse_1/strided_slice/stack?
.graph_attention_sparse_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse_1/strided_slice/stack_1?
.graph_attention_sparse_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.graph_attention_sparse_1/strided_slice/stack_2?
&graph_attention_sparse_1/strided_sliceStridedSlice+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse_1/strided_slice/stack:output:07graph_attention_sparse_1/strided_slice/stack_1:output:07graph_attention_sparse_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&graph_attention_sparse_1/strided_slice?
&graph_attention_sparse_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&graph_attention_sparse_1/GatherV2/axis?
!graph_attention_sparse_1/GatherV2GatherV2)graph_attention_sparse_1/Reshape:output:0/graph_attention_sparse_1/strided_slice:output:0/graph_attention_sparse_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2#
!graph_attention_sparse_1/GatherV2?
(graph_attention_sparse_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(graph_attention_sparse_1/Reshape_1/shape?
"graph_attention_sparse_1/Reshape_1Reshape+graph_attention_sparse_1/MatMul_2:product:01graph_attention_sparse_1/Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2$
"graph_attention_sparse_1/Reshape_1?
.graph_attention_sparse_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.graph_attention_sparse_1/strided_slice_1/stack?
0graph_attention_sparse_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0graph_attention_sparse_1/strided_slice_1/stack_1?
0graph_attention_sparse_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0graph_attention_sparse_1/strided_slice_1/stack_2?
(graph_attention_sparse_1/strided_slice_1StridedSlice+squeezed_sparse_conversion/Squeeze:output:07graph_attention_sparse_1/strided_slice_1/stack:output:09graph_attention_sparse_1/strided_slice_1/stack_1:output:09graph_attention_sparse_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2*
(graph_attention_sparse_1/strided_slice_1?
(graph_attention_sparse_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(graph_attention_sparse_1/GatherV2_1/axis?
#graph_attention_sparse_1/GatherV2_1GatherV2+graph_attention_sparse_1/Reshape_1:output:01graph_attention_sparse_1/strided_slice_1:output:01graph_attention_sparse_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2%
#graph_attention_sparse_1/GatherV2_1?
graph_attention_sparse_1/addAddV2*graph_attention_sparse_1/GatherV2:output:0,graph_attention_sparse_1/GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
graph_attention_sparse_1/add?
0graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu	LeakyRelu graph_attention_sparse_1/add:z:0*#
_output_shapes
:?????????22
0graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu?
1graph_attention_sparse_1/dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @23
1graph_attention_sparse_1/dropout_16/dropout/Const?
/graph_attention_sparse_1/dropout_16/dropout/MulMul)graph_attention_sparse_1/MatMul:product:0:graph_attention_sparse_1/dropout_16/dropout/Const:output:0*
T0*
_output_shapes
:	?21
/graph_attention_sparse_1/dropout_16/dropout/Mul?
1graph_attention_sparse_1/dropout_16/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     23
1graph_attention_sparse_1/dropout_16/dropout/Shape?
Hgraph_attention_sparse_1/dropout_16/dropout/random_uniform/RandomUniformRandomUniform:graph_attention_sparse_1/dropout_16/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02J
Hgraph_attention_sparse_1/dropout_16/dropout/random_uniform/RandomUniform?
:graph_attention_sparse_1/dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2<
:graph_attention_sparse_1/dropout_16/dropout/GreaterEqual/y?
8graph_attention_sparse_1/dropout_16/dropout/GreaterEqualGreaterEqualQgraph_attention_sparse_1/dropout_16/dropout/random_uniform/RandomUniform:output:0Cgraph_attention_sparse_1/dropout_16/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2:
8graph_attention_sparse_1/dropout_16/dropout/GreaterEqual?
0graph_attention_sparse_1/dropout_16/dropout/CastCast<graph_attention_sparse_1/dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?22
0graph_attention_sparse_1/dropout_16/dropout/Cast?
1graph_attention_sparse_1/dropout_16/dropout/Mul_1Mul3graph_attention_sparse_1/dropout_16/dropout/Mul:z:04graph_attention_sparse_1/dropout_16/dropout/Cast:y:0*
T0*
_output_shapes
:	?23
1graph_attention_sparse_1/dropout_16/dropout/Mul_1?
1graph_attention_sparse_1/dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @23
1graph_attention_sparse_1/dropout_17/dropout/Const?
/graph_attention_sparse_1/dropout_17/dropout/MulMul>graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu:activations:0:graph_attention_sparse_1/dropout_17/dropout/Const:output:0*
T0*#
_output_shapes
:?????????21
/graph_attention_sparse_1/dropout_17/dropout/Mul?
1graph_attention_sparse_1/dropout_17/dropout/ShapeShape>graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu:activations:0*
T0*
_output_shapes
:23
1graph_attention_sparse_1/dropout_17/dropout/Shape?
Hgraph_attention_sparse_1/dropout_17/dropout/random_uniform/RandomUniformRandomUniform:graph_attention_sparse_1/dropout_17/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02J
Hgraph_attention_sparse_1/dropout_17/dropout/random_uniform/RandomUniform?
:graph_attention_sparse_1/dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2<
:graph_attention_sparse_1/dropout_17/dropout/GreaterEqual/y?
8graph_attention_sparse_1/dropout_17/dropout/GreaterEqualGreaterEqualQgraph_attention_sparse_1/dropout_17/dropout/random_uniform/RandomUniform:output:0Cgraph_attention_sparse_1/dropout_17/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2:
8graph_attention_sparse_1/dropout_17/dropout/GreaterEqual?
0graph_attention_sparse_1/dropout_17/dropout/CastCast<graph_attention_sparse_1/dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????22
0graph_attention_sparse_1/dropout_17/dropout/Cast?
1graph_attention_sparse_1/dropout_17/dropout/Mul_1Mul3graph_attention_sparse_1/dropout_17/dropout/Mul:z:04graph_attention_sparse_1/dropout_17/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????23
1graph_attention_sparse_1/dropout_17/dropout/Mul_1?
1graph_attention_sparse_1/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      23
1graph_attention_sparse_1/SparseTensor/dense_shape?
4graph_attention_sparse_1/SparseSoftmax/SparseSoftmaxSparseSoftmax+squeezed_sparse_conversion/Squeeze:output:05graph_attention_sparse_1/dropout_17/dropout/Mul_1:z:0:graph_attention_sparse_1/SparseTensor/dense_shape:output:0*
T0*#
_output_shapes
:?????????26
4graph_attention_sparse_1/SparseSoftmax/SparseSoftmax?
Hgraph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0=graph_attention_sparse_1/SparseSoftmax/SparseSoftmax:output:0:graph_attention_sparse_1/SparseTensor/dense_shape:output:05graph_attention_sparse_1/dropout_16/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?2J
Hgraph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
/graph_attention_sparse_1/BiasAdd/ReadVariableOpReadVariableOp8graph_attention_sparse_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/graph_attention_sparse_1/BiasAdd/ReadVariableOp?
 graph_attention_sparse_1/BiasAddBiasAddRgraph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:07graph_attention_sparse_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse_1/BiasAdd?
graph_attention_sparse_1/stackPack)graph_attention_sparse_1/BiasAdd:output:0*
N*
T0*#
_output_shapes
:?2 
graph_attention_sparse_1/stack?
/graph_attention_sparse_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 21
/graph_attention_sparse_1/Mean/reduction_indices?
graph_attention_sparse_1/MeanMean'graph_attention_sparse_1/stack:output:08graph_attention_sparse_1/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	?2
graph_attention_sparse_1/Mean?
 graph_attention_sparse_1/SoftmaxSoftmax&graph_attention_sparse_1/Mean:output:0*
T0*
_output_shapes
:	?2"
 graph_attention_sparse_1/Softmax?
'graph_attention_sparse_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'graph_attention_sparse_1/ExpandDims/dim?
#graph_attention_sparse_1/ExpandDims
ExpandDims*graph_attention_sparse_1/Softmax:softmax:00graph_attention_sparse_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:?2%
#graph_attention_sparse_1/ExpandDims~
gather_indices/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
gather_indices/GatherV2/axis?
gather_indices/GatherV2GatherV2,graph_attention_sparse_1/ExpandDims:output:0inputs_1%gather_indices/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????*

batch_dims2
gather_indices/GatherV2?
IdentityIdentity gather_indices/GatherV2:output:0.^graph_attention_sparse/BiasAdd/ReadVariableOp0^graph_attention_sparse/BiasAdd_1/ReadVariableOp0^graph_attention_sparse/BiasAdd_2/ReadVariableOp0^graph_attention_sparse/BiasAdd_3/ReadVariableOp0^graph_attention_sparse/BiasAdd_4/ReadVariableOp0^graph_attention_sparse/BiasAdd_5/ReadVariableOp0^graph_attention_sparse/BiasAdd_6/ReadVariableOp0^graph_attention_sparse/BiasAdd_7/ReadVariableOp-^graph_attention_sparse/MatMul/ReadVariableOp/^graph_attention_sparse/MatMul_1/ReadVariableOp0^graph_attention_sparse/MatMul_10/ReadVariableOp0^graph_attention_sparse/MatMul_11/ReadVariableOp0^graph_attention_sparse/MatMul_12/ReadVariableOp0^graph_attention_sparse/MatMul_13/ReadVariableOp0^graph_attention_sparse/MatMul_14/ReadVariableOp0^graph_attention_sparse/MatMul_15/ReadVariableOp0^graph_attention_sparse/MatMul_16/ReadVariableOp0^graph_attention_sparse/MatMul_17/ReadVariableOp0^graph_attention_sparse/MatMul_18/ReadVariableOp0^graph_attention_sparse/MatMul_19/ReadVariableOp/^graph_attention_sparse/MatMul_2/ReadVariableOp0^graph_attention_sparse/MatMul_20/ReadVariableOp0^graph_attention_sparse/MatMul_21/ReadVariableOp0^graph_attention_sparse/MatMul_22/ReadVariableOp0^graph_attention_sparse/MatMul_23/ReadVariableOp/^graph_attention_sparse/MatMul_3/ReadVariableOp/^graph_attention_sparse/MatMul_4/ReadVariableOp/^graph_attention_sparse/MatMul_5/ReadVariableOp/^graph_attention_sparse/MatMul_6/ReadVariableOp/^graph_attention_sparse/MatMul_7/ReadVariableOp/^graph_attention_sparse/MatMul_8/ReadVariableOp/^graph_attention_sparse/MatMul_9/ReadVariableOp0^graph_attention_sparse_1/BiasAdd/ReadVariableOp/^graph_attention_sparse_1/MatMul/ReadVariableOp1^graph_attention_sparse_1/MatMul_1/ReadVariableOp1^graph_attention_sparse_1/MatMul_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::2^
-graph_attention_sparse/BiasAdd/ReadVariableOp-graph_attention_sparse/BiasAdd/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_1/ReadVariableOp/graph_attention_sparse/BiasAdd_1/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_2/ReadVariableOp/graph_attention_sparse/BiasAdd_2/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_3/ReadVariableOp/graph_attention_sparse/BiasAdd_3/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_4/ReadVariableOp/graph_attention_sparse/BiasAdd_4/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_5/ReadVariableOp/graph_attention_sparse/BiasAdd_5/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_6/ReadVariableOp/graph_attention_sparse/BiasAdd_6/ReadVariableOp2b
/graph_attention_sparse/BiasAdd_7/ReadVariableOp/graph_attention_sparse/BiasAdd_7/ReadVariableOp2\
,graph_attention_sparse/MatMul/ReadVariableOp,graph_attention_sparse/MatMul/ReadVariableOp2`
.graph_attention_sparse/MatMul_1/ReadVariableOp.graph_attention_sparse/MatMul_1/ReadVariableOp2b
/graph_attention_sparse/MatMul_10/ReadVariableOp/graph_attention_sparse/MatMul_10/ReadVariableOp2b
/graph_attention_sparse/MatMul_11/ReadVariableOp/graph_attention_sparse/MatMul_11/ReadVariableOp2b
/graph_attention_sparse/MatMul_12/ReadVariableOp/graph_attention_sparse/MatMul_12/ReadVariableOp2b
/graph_attention_sparse/MatMul_13/ReadVariableOp/graph_attention_sparse/MatMul_13/ReadVariableOp2b
/graph_attention_sparse/MatMul_14/ReadVariableOp/graph_attention_sparse/MatMul_14/ReadVariableOp2b
/graph_attention_sparse/MatMul_15/ReadVariableOp/graph_attention_sparse/MatMul_15/ReadVariableOp2b
/graph_attention_sparse/MatMul_16/ReadVariableOp/graph_attention_sparse/MatMul_16/ReadVariableOp2b
/graph_attention_sparse/MatMul_17/ReadVariableOp/graph_attention_sparse/MatMul_17/ReadVariableOp2b
/graph_attention_sparse/MatMul_18/ReadVariableOp/graph_attention_sparse/MatMul_18/ReadVariableOp2b
/graph_attention_sparse/MatMul_19/ReadVariableOp/graph_attention_sparse/MatMul_19/ReadVariableOp2`
.graph_attention_sparse/MatMul_2/ReadVariableOp.graph_attention_sparse/MatMul_2/ReadVariableOp2b
/graph_attention_sparse/MatMul_20/ReadVariableOp/graph_attention_sparse/MatMul_20/ReadVariableOp2b
/graph_attention_sparse/MatMul_21/ReadVariableOp/graph_attention_sparse/MatMul_21/ReadVariableOp2b
/graph_attention_sparse/MatMul_22/ReadVariableOp/graph_attention_sparse/MatMul_22/ReadVariableOp2b
/graph_attention_sparse/MatMul_23/ReadVariableOp/graph_attention_sparse/MatMul_23/ReadVariableOp2`
.graph_attention_sparse/MatMul_3/ReadVariableOp.graph_attention_sparse/MatMul_3/ReadVariableOp2`
.graph_attention_sparse/MatMul_4/ReadVariableOp.graph_attention_sparse/MatMul_4/ReadVariableOp2`
.graph_attention_sparse/MatMul_5/ReadVariableOp.graph_attention_sparse/MatMul_5/ReadVariableOp2`
.graph_attention_sparse/MatMul_6/ReadVariableOp.graph_attention_sparse/MatMul_6/ReadVariableOp2`
.graph_attention_sparse/MatMul_7/ReadVariableOp.graph_attention_sparse/MatMul_7/ReadVariableOp2`
.graph_attention_sparse/MatMul_8/ReadVariableOp.graph_attention_sparse/MatMul_8/ReadVariableOp2`
.graph_attention_sparse/MatMul_9/ReadVariableOp.graph_attention_sparse/MatMul_9/ReadVariableOp2b
/graph_attention_sparse_1/BiasAdd/ReadVariableOp/graph_attention_sparse_1/BiasAdd/ReadVariableOp2`
.graph_attention_sparse_1/MatMul/ReadVariableOp.graph_attention_sparse_1/MatMul/ReadVariableOp2d
0graph_attention_sparse_1/MatMul_1/ReadVariableOp0graph_attention_sparse_1/MatMul_1/ReadVariableOp2d
0graph_attention_sparse_1/MatMul_2/ReadVariableOp0graph_attention_sparse_1/MatMul_2/ReadVariableOp:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3
??
?#
__inference__wrapped_model_8420
input_1
input_2
input_3	
input_4?
;model_graph_attention_sparse_matmul_readvariableop_resourceA
=model_graph_attention_sparse_matmul_1_readvariableop_resourceA
=model_graph_attention_sparse_matmul_2_readvariableop_resource@
<model_graph_attention_sparse_biasadd_readvariableop_resourceA
=model_graph_attention_sparse_matmul_3_readvariableop_resourceA
=model_graph_attention_sparse_matmul_4_readvariableop_resourceA
=model_graph_attention_sparse_matmul_5_readvariableop_resourceB
>model_graph_attention_sparse_biasadd_1_readvariableop_resourceA
=model_graph_attention_sparse_matmul_6_readvariableop_resourceA
=model_graph_attention_sparse_matmul_7_readvariableop_resourceA
=model_graph_attention_sparse_matmul_8_readvariableop_resourceB
>model_graph_attention_sparse_biasadd_2_readvariableop_resourceA
=model_graph_attention_sparse_matmul_9_readvariableop_resourceB
>model_graph_attention_sparse_matmul_10_readvariableop_resourceB
>model_graph_attention_sparse_matmul_11_readvariableop_resourceB
>model_graph_attention_sparse_biasadd_3_readvariableop_resourceB
>model_graph_attention_sparse_matmul_12_readvariableop_resourceB
>model_graph_attention_sparse_matmul_13_readvariableop_resourceB
>model_graph_attention_sparse_matmul_14_readvariableop_resourceB
>model_graph_attention_sparse_biasadd_4_readvariableop_resourceB
>model_graph_attention_sparse_matmul_15_readvariableop_resourceB
>model_graph_attention_sparse_matmul_16_readvariableop_resourceB
>model_graph_attention_sparse_matmul_17_readvariableop_resourceB
>model_graph_attention_sparse_biasadd_5_readvariableop_resourceB
>model_graph_attention_sparse_matmul_18_readvariableop_resourceB
>model_graph_attention_sparse_matmul_19_readvariableop_resourceB
>model_graph_attention_sparse_matmul_20_readvariableop_resourceB
>model_graph_attention_sparse_biasadd_6_readvariableop_resourceB
>model_graph_attention_sparse_matmul_21_readvariableop_resourceB
>model_graph_attention_sparse_matmul_22_readvariableop_resourceB
>model_graph_attention_sparse_matmul_23_readvariableop_resourceB
>model_graph_attention_sparse_biasadd_7_readvariableop_resourceA
=model_graph_attention_sparse_1_matmul_readvariableop_resourceC
?model_graph_attention_sparse_1_matmul_1_readvariableop_resourceC
?model_graph_attention_sparse_1_matmul_2_readvariableop_resourceB
>model_graph_attention_sparse_1_biasadd_readvariableop_resource
identity??3model/graph_attention_sparse/BiasAdd/ReadVariableOp?5model/graph_attention_sparse/BiasAdd_1/ReadVariableOp?5model/graph_attention_sparse/BiasAdd_2/ReadVariableOp?5model/graph_attention_sparse/BiasAdd_3/ReadVariableOp?5model/graph_attention_sparse/BiasAdd_4/ReadVariableOp?5model/graph_attention_sparse/BiasAdd_5/ReadVariableOp?5model/graph_attention_sparse/BiasAdd_6/ReadVariableOp?5model/graph_attention_sparse/BiasAdd_7/ReadVariableOp?2model/graph_attention_sparse/MatMul/ReadVariableOp?4model/graph_attention_sparse/MatMul_1/ReadVariableOp?5model/graph_attention_sparse/MatMul_10/ReadVariableOp?5model/graph_attention_sparse/MatMul_11/ReadVariableOp?5model/graph_attention_sparse/MatMul_12/ReadVariableOp?5model/graph_attention_sparse/MatMul_13/ReadVariableOp?5model/graph_attention_sparse/MatMul_14/ReadVariableOp?5model/graph_attention_sparse/MatMul_15/ReadVariableOp?5model/graph_attention_sparse/MatMul_16/ReadVariableOp?5model/graph_attention_sparse/MatMul_17/ReadVariableOp?5model/graph_attention_sparse/MatMul_18/ReadVariableOp?5model/graph_attention_sparse/MatMul_19/ReadVariableOp?4model/graph_attention_sparse/MatMul_2/ReadVariableOp?5model/graph_attention_sparse/MatMul_20/ReadVariableOp?5model/graph_attention_sparse/MatMul_21/ReadVariableOp?5model/graph_attention_sparse/MatMul_22/ReadVariableOp?5model/graph_attention_sparse/MatMul_23/ReadVariableOp?4model/graph_attention_sparse/MatMul_3/ReadVariableOp?4model/graph_attention_sparse/MatMul_4/ReadVariableOp?4model/graph_attention_sparse/MatMul_5/ReadVariableOp?4model/graph_attention_sparse/MatMul_6/ReadVariableOp?4model/graph_attention_sparse/MatMul_7/ReadVariableOp?4model/graph_attention_sparse/MatMul_8/ReadVariableOp?4model/graph_attention_sparse/MatMul_9/ReadVariableOp?5model/graph_attention_sparse_1/BiasAdd/ReadVariableOp?4model/graph_attention_sparse_1/MatMul/ReadVariableOp?6model/graph_attention_sparse_1/MatMul_1/ReadVariableOp?6model/graph_attention_sparse_1/MatMul_2/ReadVariableOp?
(model/squeezed_sparse_conversion/SqueezeSqueezeinput_3*
T0	*'
_output_shapes
:?????????*
squeeze_dims
 2*
(model/squeezed_sparse_conversion/Squeeze?
*model/squeezed_sparse_conversion/Squeeze_1Squeezeinput_4*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2,
*model/squeezed_sparse_conversion/Squeeze_1?
9model/squeezed_sparse_conversion/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2;
9model/squeezed_sparse_conversion/SparseTensor/dense_shapet
model/dropout/IdentityIdentityinput_1*
T0*$
_output_shapes
:??2
model/dropout/Identity?
$model/graph_attention_sparse/SqueezeSqueezemodel/dropout/Identity:output:0*
T0* 
_output_shapes
:
??*
squeeze_dims
 2&
$model/graph_attention_sparse/Squeeze?
2model/graph_attention_sparse/MatMul/ReadVariableOpReadVariableOp;model_graph_attention_sparse_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype024
2model/graph_attention_sparse/MatMul/ReadVariableOp?
#model/graph_attention_sparse/MatMulMatMul-model/graph_attention_sparse/Squeeze:output:0:model/graph_attention_sparse/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2%
#model/graph_attention_sparse/MatMul?
4model/graph_attention_sparse/MatMul_1/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype026
4model/graph_attention_sparse/MatMul_1/ReadVariableOp?
%model/graph_attention_sparse/MatMul_1MatMul-model/graph_attention_sparse/MatMul:product:0<model/graph_attention_sparse/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_1?
4model/graph_attention_sparse/MatMul_2/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype026
4model/graph_attention_sparse/MatMul_2/ReadVariableOp?
%model/graph_attention_sparse/MatMul_2MatMul-model/graph_attention_sparse/MatMul:product:0<model/graph_attention_sparse/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_2?
*model/graph_attention_sparse/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*model/graph_attention_sparse/Reshape/shape?
$model/graph_attention_sparse/ReshapeReshape/model/graph_attention_sparse/MatMul_1:product:03model/graph_attention_sparse/Reshape/shape:output:0*
T0*
_output_shapes	
:?2&
$model/graph_attention_sparse/Reshape?
0model/graph_attention_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0model/graph_attention_sparse/strided_slice/stack?
2model/graph_attention_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2model/graph_attention_sparse/strided_slice/stack_1?
2model/graph_attention_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2model/graph_attention_sparse/strided_slice/stack_2?
*model/graph_attention_sparse/strided_sliceStridedSlice1model/squeezed_sparse_conversion/Squeeze:output:09model/graph_attention_sparse/strided_slice/stack:output:0;model/graph_attention_sparse/strided_slice/stack_1:output:0;model/graph_attention_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2,
*model/graph_attention_sparse/strided_slice?
*model/graph_attention_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model/graph_attention_sparse/GatherV2/axis?
%model/graph_attention_sparse/GatherV2GatherV2-model/graph_attention_sparse/Reshape:output:03model/graph_attention_sparse/strided_slice:output:03model/graph_attention_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2'
%model/graph_attention_sparse/GatherV2?
,model/graph_attention_sparse/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_1/shape?
&model/graph_attention_sparse/Reshape_1Reshape/model/graph_attention_sparse/MatMul_2:product:05model/graph_attention_sparse/Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_1?
2model/graph_attention_sparse/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       24
2model/graph_attention_sparse/strided_slice_1/stack?
4model/graph_attention_sparse/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_1/stack_1?
4model/graph_attention_sparse/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_1/stack_2?
,model/graph_attention_sparse/strided_slice_1StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_1/stack:output:0=model/graph_attention_sparse/strided_slice_1/stack_1:output:0=model/graph_attention_sparse/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_1?
,model/graph_attention_sparse/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_1/axis?
'model/graph_attention_sparse/GatherV2_1GatherV2/model/graph_attention_sparse/Reshape_1:output:05model/graph_attention_sparse/strided_slice_1:output:05model/graph_attention_sparse/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_1?
 model/graph_attention_sparse/addAddV2.model/graph_attention_sparse/GatherV2:output:00model/graph_attention_sparse/GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2"
 model/graph_attention_sparse/add?
2model/graph_attention_sparse/leaky_re_lu/LeakyRelu	LeakyRelu$model/graph_attention_sparse/add:z:0*#
_output_shapes
:?????????24
2model/graph_attention_sparse/leaky_re_lu/LeakyRelu?
-model/graph_attention_sparse/dropout/IdentityIdentity-model/graph_attention_sparse/MatMul:product:0*
T0*
_output_shapes
:	?2/
-model/graph_attention_sparse/dropout/Identity?
/model/graph_attention_sparse/dropout_1/IdentityIdentity@model/graph_attention_sparse/leaky_re_lu/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????21
/model/graph_attention_sparse/dropout_1/Identity?
5model/graph_attention_sparse/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      27
5model/graph_attention_sparse/SparseTensor/dense_shape?
8model/graph_attention_sparse/SparseSoftmax/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:08model/graph_attention_sparse/dropout_1/Identity:output:0>model/graph_attention_sparse/SparseTensor/dense_shape:output:0*
T0*#
_output_shapes
:?????????2:
8model/graph_attention_sparse/SparseSoftmax/SparseSoftmax?
Lmodel/graph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Amodel/graph_attention_sparse/SparseSoftmax/SparseSoftmax:output:0>model/graph_attention_sparse/SparseTensor/dense_shape:output:06model/graph_attention_sparse/dropout/Identity:output:0*
T0*
_output_shapes
:	?2N
Lmodel/graph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
3model/graph_attention_sparse/BiasAdd/ReadVariableOpReadVariableOp<model_graph_attention_sparse_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model/graph_attention_sparse/BiasAdd/ReadVariableOp?
$model/graph_attention_sparse/BiasAddBiasAddVmodel/graph_attention_sparse/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0;model/graph_attention_sparse/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2&
$model/graph_attention_sparse/BiasAdd?
4model/graph_attention_sparse/MatMul_3/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_3_readvariableop_resource*
_output_shapes
:	?*
dtype026
4model/graph_attention_sparse/MatMul_3/ReadVariableOp?
%model/graph_attention_sparse/MatMul_3MatMul-model/graph_attention_sparse/Squeeze:output:0<model/graph_attention_sparse/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_3?
4model/graph_attention_sparse/MatMul_4/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_4_readvariableop_resource*
_output_shapes

:*
dtype026
4model/graph_attention_sparse/MatMul_4/ReadVariableOp?
%model/graph_attention_sparse/MatMul_4MatMul/model/graph_attention_sparse/MatMul_3:product:0<model/graph_attention_sparse/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_4?
4model/graph_attention_sparse/MatMul_5/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_5_readvariableop_resource*
_output_shapes

:*
dtype026
4model/graph_attention_sparse/MatMul_5/ReadVariableOp?
%model/graph_attention_sparse/MatMul_5MatMul/model/graph_attention_sparse/MatMul_3:product:0<model/graph_attention_sparse/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_5?
,model/graph_attention_sparse/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_2/shape?
&model/graph_attention_sparse/Reshape_2Reshape/model/graph_attention_sparse/MatMul_4:product:05model/graph_attention_sparse/Reshape_2/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_2?
2model/graph_attention_sparse/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model/graph_attention_sparse/strided_slice_2/stack?
4model/graph_attention_sparse/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_2/stack_1?
4model/graph_attention_sparse/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_2/stack_2?
,model/graph_attention_sparse/strided_slice_2StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_2/stack:output:0=model/graph_attention_sparse/strided_slice_2/stack_1:output:0=model/graph_attention_sparse/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_2?
,model/graph_attention_sparse/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_2/axis?
'model/graph_attention_sparse/GatherV2_2GatherV2/model/graph_attention_sparse/Reshape_2:output:05model/graph_attention_sparse/strided_slice_2:output:05model/graph_attention_sparse/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_2?
,model/graph_attention_sparse/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_3/shape?
&model/graph_attention_sparse/Reshape_3Reshape/model/graph_attention_sparse/MatMul_5:product:05model/graph_attention_sparse/Reshape_3/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_3?
2model/graph_attention_sparse/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       24
2model/graph_attention_sparse/strided_slice_3/stack?
4model/graph_attention_sparse/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_3/stack_1?
4model/graph_attention_sparse/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_3/stack_2?
,model/graph_attention_sparse/strided_slice_3StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_3/stack:output:0=model/graph_attention_sparse/strided_slice_3/stack_1:output:0=model/graph_attention_sparse/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_3?
,model/graph_attention_sparse/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_3/axis?
'model/graph_attention_sparse/GatherV2_3GatherV2/model/graph_attention_sparse/Reshape_3:output:05model/graph_attention_sparse/strided_slice_3:output:05model/graph_attention_sparse/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_3?
"model/graph_attention_sparse/add_1AddV20model/graph_attention_sparse/GatherV2_2:output:00model/graph_attention_sparse/GatherV2_3:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse/add_1?
4model/graph_attention_sparse/leaky_re_lu_1/LeakyRelu	LeakyRelu&model/graph_attention_sparse/add_1:z:0*#
_output_shapes
:?????????26
4model/graph_attention_sparse/leaky_re_lu_1/LeakyRelu?
/model/graph_attention_sparse/dropout_2/IdentityIdentity/model/graph_attention_sparse/MatMul_3:product:0*
T0*
_output_shapes
:	?21
/model/graph_attention_sparse/dropout_2/Identity?
/model/graph_attention_sparse/dropout_3/IdentityIdentityBmodel/graph_attention_sparse/leaky_re_lu_1/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????21
/model/graph_attention_sparse/dropout_3/Identity?
7model/graph_attention_sparse/SparseTensor_1/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse/SparseTensor_1/dense_shape?
:model/graph_attention_sparse/SparseSoftmax_1/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:08model/graph_attention_sparse/dropout_3/Identity:output:0@model/graph_attention_sparse/SparseTensor_1/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse/SparseSoftmax_1/SparseSoftmax?
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse/SparseSoftmax_1/SparseSoftmax:output:0@model/graph_attention_sparse/SparseTensor_1/dense_shape:output:08model/graph_attention_sparse/dropout_2/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMul?
5model/graph_attention_sparse/BiasAdd_1/ReadVariableOpReadVariableOp>model_graph_attention_sparse_biasadd_1_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse/BiasAdd_1/ReadVariableOp?
&model/graph_attention_sparse/BiasAdd_1BiasAddXmodel/graph_attention_sparse/SparseTensorDenseMatMul_1/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/BiasAdd_1?
4model/graph_attention_sparse/MatMul_6/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_6_readvariableop_resource*
_output_shapes
:	?*
dtype026
4model/graph_attention_sparse/MatMul_6/ReadVariableOp?
%model/graph_attention_sparse/MatMul_6MatMul-model/graph_attention_sparse/Squeeze:output:0<model/graph_attention_sparse/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_6?
4model/graph_attention_sparse/MatMul_7/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_7_readvariableop_resource*
_output_shapes

:*
dtype026
4model/graph_attention_sparse/MatMul_7/ReadVariableOp?
%model/graph_attention_sparse/MatMul_7MatMul/model/graph_attention_sparse/MatMul_6:product:0<model/graph_attention_sparse/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_7?
4model/graph_attention_sparse/MatMul_8/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_8_readvariableop_resource*
_output_shapes

:*
dtype026
4model/graph_attention_sparse/MatMul_8/ReadVariableOp?
%model/graph_attention_sparse/MatMul_8MatMul/model/graph_attention_sparse/MatMul_6:product:0<model/graph_attention_sparse/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_8?
,model/graph_attention_sparse/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_4/shape?
&model/graph_attention_sparse/Reshape_4Reshape/model/graph_attention_sparse/MatMul_7:product:05model/graph_attention_sparse/Reshape_4/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_4?
2model/graph_attention_sparse/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model/graph_attention_sparse/strided_slice_4/stack?
4model/graph_attention_sparse/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_4/stack_1?
4model/graph_attention_sparse/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_4/stack_2?
,model/graph_attention_sparse/strided_slice_4StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_4/stack:output:0=model/graph_attention_sparse/strided_slice_4/stack_1:output:0=model/graph_attention_sparse/strided_slice_4/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_4?
,model/graph_attention_sparse/GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_4/axis?
'model/graph_attention_sparse/GatherV2_4GatherV2/model/graph_attention_sparse/Reshape_4:output:05model/graph_attention_sparse/strided_slice_4:output:05model/graph_attention_sparse/GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_4?
,model/graph_attention_sparse/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_5/shape?
&model/graph_attention_sparse/Reshape_5Reshape/model/graph_attention_sparse/MatMul_8:product:05model/graph_attention_sparse/Reshape_5/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_5?
2model/graph_attention_sparse/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       24
2model/graph_attention_sparse/strided_slice_5/stack?
4model/graph_attention_sparse/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_5/stack_1?
4model/graph_attention_sparse/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_5/stack_2?
,model/graph_attention_sparse/strided_slice_5StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_5/stack:output:0=model/graph_attention_sparse/strided_slice_5/stack_1:output:0=model/graph_attention_sparse/strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_5?
,model/graph_attention_sparse/GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_5/axis?
'model/graph_attention_sparse/GatherV2_5GatherV2/model/graph_attention_sparse/Reshape_5:output:05model/graph_attention_sparse/strided_slice_5:output:05model/graph_attention_sparse/GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_5?
"model/graph_attention_sparse/add_2AddV20model/graph_attention_sparse/GatherV2_4:output:00model/graph_attention_sparse/GatherV2_5:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse/add_2?
4model/graph_attention_sparse/leaky_re_lu_2/LeakyRelu	LeakyRelu&model/graph_attention_sparse/add_2:z:0*#
_output_shapes
:?????????26
4model/graph_attention_sparse/leaky_re_lu_2/LeakyRelu?
/model/graph_attention_sparse/dropout_4/IdentityIdentity/model/graph_attention_sparse/MatMul_6:product:0*
T0*
_output_shapes
:	?21
/model/graph_attention_sparse/dropout_4/Identity?
/model/graph_attention_sparse/dropout_5/IdentityIdentityBmodel/graph_attention_sparse/leaky_re_lu_2/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????21
/model/graph_attention_sparse/dropout_5/Identity?
7model/graph_attention_sparse/SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse/SparseTensor_2/dense_shape?
:model/graph_attention_sparse/SparseSoftmax_2/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:08model/graph_attention_sparse/dropout_5/Identity:output:0@model/graph_attention_sparse/SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse/SparseSoftmax_2/SparseSoftmax?
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse/SparseSoftmax_2/SparseSoftmax:output:0@model/graph_attention_sparse/SparseTensor_2/dense_shape:output:08model/graph_attention_sparse/dropout_4/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMul?
5model/graph_attention_sparse/BiasAdd_2/ReadVariableOpReadVariableOp>model_graph_attention_sparse_biasadd_2_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse/BiasAdd_2/ReadVariableOp?
&model/graph_attention_sparse/BiasAdd_2BiasAddXmodel/graph_attention_sparse/SparseTensorDenseMatMul_2/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/BiasAdd_2?
4model/graph_attention_sparse/MatMul_9/ReadVariableOpReadVariableOp=model_graph_attention_sparse_matmul_9_readvariableop_resource*
_output_shapes
:	?*
dtype026
4model/graph_attention_sparse/MatMul_9/ReadVariableOp?
%model/graph_attention_sparse/MatMul_9MatMul-model/graph_attention_sparse/Squeeze:output:0<model/graph_attention_sparse/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse/MatMul_9?
5model/graph_attention_sparse/MatMul_10/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_10_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_10/ReadVariableOp?
&model/graph_attention_sparse/MatMul_10MatMul/model/graph_attention_sparse/MatMul_9:product:0=model/graph_attention_sparse/MatMul_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_10?
5model/graph_attention_sparse/MatMul_11/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_11_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_11/ReadVariableOp?
&model/graph_attention_sparse/MatMul_11MatMul/model/graph_attention_sparse/MatMul_9:product:0=model/graph_attention_sparse/MatMul_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_11?
,model/graph_attention_sparse/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_6/shape?
&model/graph_attention_sparse/Reshape_6Reshape0model/graph_attention_sparse/MatMul_10:product:05model/graph_attention_sparse/Reshape_6/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_6?
2model/graph_attention_sparse/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model/graph_attention_sparse/strided_slice_6/stack?
4model/graph_attention_sparse/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_6/stack_1?
4model/graph_attention_sparse/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_6/stack_2?
,model/graph_attention_sparse/strided_slice_6StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_6/stack:output:0=model/graph_attention_sparse/strided_slice_6/stack_1:output:0=model/graph_attention_sparse/strided_slice_6/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_6?
,model/graph_attention_sparse/GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_6/axis?
'model/graph_attention_sparse/GatherV2_6GatherV2/model/graph_attention_sparse/Reshape_6:output:05model/graph_attention_sparse/strided_slice_6:output:05model/graph_attention_sparse/GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_6?
,model/graph_attention_sparse/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_7/shape?
&model/graph_attention_sparse/Reshape_7Reshape0model/graph_attention_sparse/MatMul_11:product:05model/graph_attention_sparse/Reshape_7/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_7?
2model/graph_attention_sparse/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       24
2model/graph_attention_sparse/strided_slice_7/stack?
4model/graph_attention_sparse/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_7/stack_1?
4model/graph_attention_sparse/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_7/stack_2?
,model/graph_attention_sparse/strided_slice_7StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_7/stack:output:0=model/graph_attention_sparse/strided_slice_7/stack_1:output:0=model/graph_attention_sparse/strided_slice_7/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_7?
,model/graph_attention_sparse/GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_7/axis?
'model/graph_attention_sparse/GatherV2_7GatherV2/model/graph_attention_sparse/Reshape_7:output:05model/graph_attention_sparse/strided_slice_7:output:05model/graph_attention_sparse/GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_7?
"model/graph_attention_sparse/add_3AddV20model/graph_attention_sparse/GatherV2_6:output:00model/graph_attention_sparse/GatherV2_7:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse/add_3?
4model/graph_attention_sparse/leaky_re_lu_3/LeakyRelu	LeakyRelu&model/graph_attention_sparse/add_3:z:0*#
_output_shapes
:?????????26
4model/graph_attention_sparse/leaky_re_lu_3/LeakyRelu?
/model/graph_attention_sparse/dropout_6/IdentityIdentity/model/graph_attention_sparse/MatMul_9:product:0*
T0*
_output_shapes
:	?21
/model/graph_attention_sparse/dropout_6/Identity?
/model/graph_attention_sparse/dropout_7/IdentityIdentityBmodel/graph_attention_sparse/leaky_re_lu_3/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????21
/model/graph_attention_sparse/dropout_7/Identity?
7model/graph_attention_sparse/SparseTensor_3/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse/SparseTensor_3/dense_shape?
:model/graph_attention_sparse/SparseSoftmax_3/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:08model/graph_attention_sparse/dropout_7/Identity:output:0@model/graph_attention_sparse/SparseTensor_3/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse/SparseSoftmax_3/SparseSoftmax?
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse/SparseSoftmax_3/SparseSoftmax:output:0@model/graph_attention_sparse/SparseTensor_3/dense_shape:output:08model/graph_attention_sparse/dropout_6/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMul?
5model/graph_attention_sparse/BiasAdd_3/ReadVariableOpReadVariableOp>model_graph_attention_sparse_biasadd_3_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse/BiasAdd_3/ReadVariableOp?
&model/graph_attention_sparse/BiasAdd_3BiasAddXmodel/graph_attention_sparse/SparseTensorDenseMatMul_3/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/BiasAdd_3?
5model/graph_attention_sparse/MatMul_12/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_12_readvariableop_resource*
_output_shapes
:	?*
dtype027
5model/graph_attention_sparse/MatMul_12/ReadVariableOp?
&model/graph_attention_sparse/MatMul_12MatMul-model/graph_attention_sparse/Squeeze:output:0=model/graph_attention_sparse/MatMul_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_12?
5model/graph_attention_sparse/MatMul_13/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_13_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_13/ReadVariableOp?
&model/graph_attention_sparse/MatMul_13MatMul0model/graph_attention_sparse/MatMul_12:product:0=model/graph_attention_sparse/MatMul_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_13?
5model/graph_attention_sparse/MatMul_14/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_14_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_14/ReadVariableOp?
&model/graph_attention_sparse/MatMul_14MatMul0model/graph_attention_sparse/MatMul_12:product:0=model/graph_attention_sparse/MatMul_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_14?
,model/graph_attention_sparse/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_8/shape?
&model/graph_attention_sparse/Reshape_8Reshape0model/graph_attention_sparse/MatMul_13:product:05model/graph_attention_sparse/Reshape_8/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_8?
2model/graph_attention_sparse/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model/graph_attention_sparse/strided_slice_8/stack?
4model/graph_attention_sparse/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_8/stack_1?
4model/graph_attention_sparse/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_8/stack_2?
,model/graph_attention_sparse/strided_slice_8StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_8/stack:output:0=model/graph_attention_sparse/strided_slice_8/stack_1:output:0=model/graph_attention_sparse/strided_slice_8/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_8?
,model/graph_attention_sparse/GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_8/axis?
'model/graph_attention_sparse/GatherV2_8GatherV2/model/graph_attention_sparse/Reshape_8:output:05model/graph_attention_sparse/strided_slice_8:output:05model/graph_attention_sparse/GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_8?
,model/graph_attention_sparse/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse/Reshape_9/shape?
&model/graph_attention_sparse/Reshape_9Reshape0model/graph_attention_sparse/MatMul_14:product:05model/graph_attention_sparse/Reshape_9/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse/Reshape_9?
2model/graph_attention_sparse/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       24
2model/graph_attention_sparse/strided_slice_9/stack?
4model/graph_attention_sparse/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse/strided_slice_9/stack_1?
4model/graph_attention_sparse/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse/strided_slice_9/stack_2?
,model/graph_attention_sparse/strided_slice_9StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse/strided_slice_9/stack:output:0=model/graph_attention_sparse/strided_slice_9/stack_1:output:0=model/graph_attention_sparse/strided_slice_9/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse/strided_slice_9?
,model/graph_attention_sparse/GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse/GatherV2_9/axis?
'model/graph_attention_sparse/GatherV2_9GatherV2/model/graph_attention_sparse/Reshape_9:output:05model/graph_attention_sparse/strided_slice_9:output:05model/graph_attention_sparse/GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse/GatherV2_9?
"model/graph_attention_sparse/add_4AddV20model/graph_attention_sparse/GatherV2_8:output:00model/graph_attention_sparse/GatherV2_9:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse/add_4?
4model/graph_attention_sparse/leaky_re_lu_4/LeakyRelu	LeakyRelu&model/graph_attention_sparse/add_4:z:0*#
_output_shapes
:?????????26
4model/graph_attention_sparse/leaky_re_lu_4/LeakyRelu?
/model/graph_attention_sparse/dropout_8/IdentityIdentity0model/graph_attention_sparse/MatMul_12:product:0*
T0*
_output_shapes
:	?21
/model/graph_attention_sparse/dropout_8/Identity?
/model/graph_attention_sparse/dropout_9/IdentityIdentityBmodel/graph_attention_sparse/leaky_re_lu_4/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????21
/model/graph_attention_sparse/dropout_9/Identity?
7model/graph_attention_sparse/SparseTensor_4/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse/SparseTensor_4/dense_shape?
:model/graph_attention_sparse/SparseSoftmax_4/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:08model/graph_attention_sparse/dropout_9/Identity:output:0@model/graph_attention_sparse/SparseTensor_4/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse/SparseSoftmax_4/SparseSoftmax?
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse/SparseSoftmax_4/SparseSoftmax:output:0@model/graph_attention_sparse/SparseTensor_4/dense_shape:output:08model/graph_attention_sparse/dropout_8/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMul?
5model/graph_attention_sparse/BiasAdd_4/ReadVariableOpReadVariableOp>model_graph_attention_sparse_biasadd_4_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse/BiasAdd_4/ReadVariableOp?
&model/graph_attention_sparse/BiasAdd_4BiasAddXmodel/graph_attention_sparse/SparseTensorDenseMatMul_4/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/BiasAdd_4?
5model/graph_attention_sparse/MatMul_15/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_15_readvariableop_resource*
_output_shapes
:	?*
dtype027
5model/graph_attention_sparse/MatMul_15/ReadVariableOp?
&model/graph_attention_sparse/MatMul_15MatMul-model/graph_attention_sparse/Squeeze:output:0=model/graph_attention_sparse/MatMul_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_15?
5model/graph_attention_sparse/MatMul_16/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_16_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_16/ReadVariableOp?
&model/graph_attention_sparse/MatMul_16MatMul0model/graph_attention_sparse/MatMul_15:product:0=model/graph_attention_sparse/MatMul_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_16?
5model/graph_attention_sparse/MatMul_17/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_17_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_17/ReadVariableOp?
&model/graph_attention_sparse/MatMul_17MatMul0model/graph_attention_sparse/MatMul_15:product:0=model/graph_attention_sparse/MatMul_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_17?
-model/graph_attention_sparse/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-model/graph_attention_sparse/Reshape_10/shape?
'model/graph_attention_sparse/Reshape_10Reshape0model/graph_attention_sparse/MatMul_16:product:06model/graph_attention_sparse/Reshape_10/shape:output:0*
T0*
_output_shapes	
:?2)
'model/graph_attention_sparse/Reshape_10?
3model/graph_attention_sparse/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3model/graph_attention_sparse/strided_slice_10/stack?
5model/graph_attention_sparse/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5model/graph_attention_sparse/strided_slice_10/stack_1?
5model/graph_attention_sparse/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5model/graph_attention_sparse/strided_slice_10/stack_2?
-model/graph_attention_sparse/strided_slice_10StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0<model/graph_attention_sparse/strided_slice_10/stack:output:0>model/graph_attention_sparse/strided_slice_10/stack_1:output:0>model/graph_attention_sparse/strided_slice_10/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2/
-model/graph_attention_sparse/strided_slice_10?
-model/graph_attention_sparse/GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model/graph_attention_sparse/GatherV2_10/axis?
(model/graph_attention_sparse/GatherV2_10GatherV20model/graph_attention_sparse/Reshape_10:output:06model/graph_attention_sparse/strided_slice_10:output:06model/graph_attention_sparse/GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2*
(model/graph_attention_sparse/GatherV2_10?
-model/graph_attention_sparse/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-model/graph_attention_sparse/Reshape_11/shape?
'model/graph_attention_sparse/Reshape_11Reshape0model/graph_attention_sparse/MatMul_17:product:06model/graph_attention_sparse/Reshape_11/shape:output:0*
T0*
_output_shapes	
:?2)
'model/graph_attention_sparse/Reshape_11?
3model/graph_attention_sparse/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       25
3model/graph_attention_sparse/strided_slice_11/stack?
5model/graph_attention_sparse/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5model/graph_attention_sparse/strided_slice_11/stack_1?
5model/graph_attention_sparse/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5model/graph_attention_sparse/strided_slice_11/stack_2?
-model/graph_attention_sparse/strided_slice_11StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0<model/graph_attention_sparse/strided_slice_11/stack:output:0>model/graph_attention_sparse/strided_slice_11/stack_1:output:0>model/graph_attention_sparse/strided_slice_11/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2/
-model/graph_attention_sparse/strided_slice_11?
-model/graph_attention_sparse/GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model/graph_attention_sparse/GatherV2_11/axis?
(model/graph_attention_sparse/GatherV2_11GatherV20model/graph_attention_sparse/Reshape_11:output:06model/graph_attention_sparse/strided_slice_11:output:06model/graph_attention_sparse/GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2*
(model/graph_attention_sparse/GatherV2_11?
"model/graph_attention_sparse/add_5AddV21model/graph_attention_sparse/GatherV2_10:output:01model/graph_attention_sparse/GatherV2_11:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse/add_5?
4model/graph_attention_sparse/leaky_re_lu_5/LeakyRelu	LeakyRelu&model/graph_attention_sparse/add_5:z:0*#
_output_shapes
:?????????26
4model/graph_attention_sparse/leaky_re_lu_5/LeakyRelu?
0model/graph_attention_sparse/dropout_10/IdentityIdentity0model/graph_attention_sparse/MatMul_15:product:0*
T0*
_output_shapes
:	?22
0model/graph_attention_sparse/dropout_10/Identity?
0model/graph_attention_sparse/dropout_11/IdentityIdentityBmodel/graph_attention_sparse/leaky_re_lu_5/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????22
0model/graph_attention_sparse/dropout_11/Identity?
7model/graph_attention_sparse/SparseTensor_5/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse/SparseTensor_5/dense_shape?
:model/graph_attention_sparse/SparseSoftmax_5/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:09model/graph_attention_sparse/dropout_11/Identity:output:0@model/graph_attention_sparse/SparseTensor_5/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse/SparseSoftmax_5/SparseSoftmax?
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse/SparseSoftmax_5/SparseSoftmax:output:0@model/graph_attention_sparse/SparseTensor_5/dense_shape:output:09model/graph_attention_sparse/dropout_10/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMul?
5model/graph_attention_sparse/BiasAdd_5/ReadVariableOpReadVariableOp>model_graph_attention_sparse_biasadd_5_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse/BiasAdd_5/ReadVariableOp?
&model/graph_attention_sparse/BiasAdd_5BiasAddXmodel/graph_attention_sparse/SparseTensorDenseMatMul_5/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/BiasAdd_5?
5model/graph_attention_sparse/MatMul_18/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_18_readvariableop_resource*
_output_shapes
:	?*
dtype027
5model/graph_attention_sparse/MatMul_18/ReadVariableOp?
&model/graph_attention_sparse/MatMul_18MatMul-model/graph_attention_sparse/Squeeze:output:0=model/graph_attention_sparse/MatMul_18/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_18?
5model/graph_attention_sparse/MatMul_19/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_19_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_19/ReadVariableOp?
&model/graph_attention_sparse/MatMul_19MatMul0model/graph_attention_sparse/MatMul_18:product:0=model/graph_attention_sparse/MatMul_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_19?
5model/graph_attention_sparse/MatMul_20/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_20_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_20/ReadVariableOp?
&model/graph_attention_sparse/MatMul_20MatMul0model/graph_attention_sparse/MatMul_18:product:0=model/graph_attention_sparse/MatMul_20/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_20?
-model/graph_attention_sparse/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-model/graph_attention_sparse/Reshape_12/shape?
'model/graph_attention_sparse/Reshape_12Reshape0model/graph_attention_sparse/MatMul_19:product:06model/graph_attention_sparse/Reshape_12/shape:output:0*
T0*
_output_shapes	
:?2)
'model/graph_attention_sparse/Reshape_12?
3model/graph_attention_sparse/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3model/graph_attention_sparse/strided_slice_12/stack?
5model/graph_attention_sparse/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5model/graph_attention_sparse/strided_slice_12/stack_1?
5model/graph_attention_sparse/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5model/graph_attention_sparse/strided_slice_12/stack_2?
-model/graph_attention_sparse/strided_slice_12StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0<model/graph_attention_sparse/strided_slice_12/stack:output:0>model/graph_attention_sparse/strided_slice_12/stack_1:output:0>model/graph_attention_sparse/strided_slice_12/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2/
-model/graph_attention_sparse/strided_slice_12?
-model/graph_attention_sparse/GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model/graph_attention_sparse/GatherV2_12/axis?
(model/graph_attention_sparse/GatherV2_12GatherV20model/graph_attention_sparse/Reshape_12:output:06model/graph_attention_sparse/strided_slice_12:output:06model/graph_attention_sparse/GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2*
(model/graph_attention_sparse/GatherV2_12?
-model/graph_attention_sparse/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-model/graph_attention_sparse/Reshape_13/shape?
'model/graph_attention_sparse/Reshape_13Reshape0model/graph_attention_sparse/MatMul_20:product:06model/graph_attention_sparse/Reshape_13/shape:output:0*
T0*
_output_shapes	
:?2)
'model/graph_attention_sparse/Reshape_13?
3model/graph_attention_sparse/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       25
3model/graph_attention_sparse/strided_slice_13/stack?
5model/graph_attention_sparse/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5model/graph_attention_sparse/strided_slice_13/stack_1?
5model/graph_attention_sparse/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5model/graph_attention_sparse/strided_slice_13/stack_2?
-model/graph_attention_sparse/strided_slice_13StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0<model/graph_attention_sparse/strided_slice_13/stack:output:0>model/graph_attention_sparse/strided_slice_13/stack_1:output:0>model/graph_attention_sparse/strided_slice_13/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2/
-model/graph_attention_sparse/strided_slice_13?
-model/graph_attention_sparse/GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model/graph_attention_sparse/GatherV2_13/axis?
(model/graph_attention_sparse/GatherV2_13GatherV20model/graph_attention_sparse/Reshape_13:output:06model/graph_attention_sparse/strided_slice_13:output:06model/graph_attention_sparse/GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2*
(model/graph_attention_sparse/GatherV2_13?
"model/graph_attention_sparse/add_6AddV21model/graph_attention_sparse/GatherV2_12:output:01model/graph_attention_sparse/GatherV2_13:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse/add_6?
4model/graph_attention_sparse/leaky_re_lu_6/LeakyRelu	LeakyRelu&model/graph_attention_sparse/add_6:z:0*#
_output_shapes
:?????????26
4model/graph_attention_sparse/leaky_re_lu_6/LeakyRelu?
0model/graph_attention_sparse/dropout_12/IdentityIdentity0model/graph_attention_sparse/MatMul_18:product:0*
T0*
_output_shapes
:	?22
0model/graph_attention_sparse/dropout_12/Identity?
0model/graph_attention_sparse/dropout_13/IdentityIdentityBmodel/graph_attention_sparse/leaky_re_lu_6/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????22
0model/graph_attention_sparse/dropout_13/Identity?
7model/graph_attention_sparse/SparseTensor_6/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse/SparseTensor_6/dense_shape?
:model/graph_attention_sparse/SparseSoftmax_6/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:09model/graph_attention_sparse/dropout_13/Identity:output:0@model/graph_attention_sparse/SparseTensor_6/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse/SparseSoftmax_6/SparseSoftmax?
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse/SparseSoftmax_6/SparseSoftmax:output:0@model/graph_attention_sparse/SparseTensor_6/dense_shape:output:09model/graph_attention_sparse/dropout_12/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMul?
5model/graph_attention_sparse/BiasAdd_6/ReadVariableOpReadVariableOp>model_graph_attention_sparse_biasadd_6_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse/BiasAdd_6/ReadVariableOp?
&model/graph_attention_sparse/BiasAdd_6BiasAddXmodel/graph_attention_sparse/SparseTensorDenseMatMul_6/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/BiasAdd_6?
5model/graph_attention_sparse/MatMul_21/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_21_readvariableop_resource*
_output_shapes
:	?*
dtype027
5model/graph_attention_sparse/MatMul_21/ReadVariableOp?
&model/graph_attention_sparse/MatMul_21MatMul-model/graph_attention_sparse/Squeeze:output:0=model/graph_attention_sparse/MatMul_21/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_21?
5model/graph_attention_sparse/MatMul_22/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_22_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_22/ReadVariableOp?
&model/graph_attention_sparse/MatMul_22MatMul0model/graph_attention_sparse/MatMul_21:product:0=model/graph_attention_sparse/MatMul_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_22?
5model/graph_attention_sparse/MatMul_23/ReadVariableOpReadVariableOp>model_graph_attention_sparse_matmul_23_readvariableop_resource*
_output_shapes

:*
dtype027
5model/graph_attention_sparse/MatMul_23/ReadVariableOp?
&model/graph_attention_sparse/MatMul_23MatMul0model/graph_attention_sparse/MatMul_21:product:0=model/graph_attention_sparse/MatMul_23/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/MatMul_23?
-model/graph_attention_sparse/Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-model/graph_attention_sparse/Reshape_14/shape?
'model/graph_attention_sparse/Reshape_14Reshape0model/graph_attention_sparse/MatMul_22:product:06model/graph_attention_sparse/Reshape_14/shape:output:0*
T0*
_output_shapes	
:?2)
'model/graph_attention_sparse/Reshape_14?
3model/graph_attention_sparse/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3model/graph_attention_sparse/strided_slice_14/stack?
5model/graph_attention_sparse/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5model/graph_attention_sparse/strided_slice_14/stack_1?
5model/graph_attention_sparse/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5model/graph_attention_sparse/strided_slice_14/stack_2?
-model/graph_attention_sparse/strided_slice_14StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0<model/graph_attention_sparse/strided_slice_14/stack:output:0>model/graph_attention_sparse/strided_slice_14/stack_1:output:0>model/graph_attention_sparse/strided_slice_14/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2/
-model/graph_attention_sparse/strided_slice_14?
-model/graph_attention_sparse/GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model/graph_attention_sparse/GatherV2_14/axis?
(model/graph_attention_sparse/GatherV2_14GatherV20model/graph_attention_sparse/Reshape_14:output:06model/graph_attention_sparse/strided_slice_14:output:06model/graph_attention_sparse/GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2*
(model/graph_attention_sparse/GatherV2_14?
-model/graph_attention_sparse/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-model/graph_attention_sparse/Reshape_15/shape?
'model/graph_attention_sparse/Reshape_15Reshape0model/graph_attention_sparse/MatMul_23:product:06model/graph_attention_sparse/Reshape_15/shape:output:0*
T0*
_output_shapes	
:?2)
'model/graph_attention_sparse/Reshape_15?
3model/graph_attention_sparse/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       25
3model/graph_attention_sparse/strided_slice_15/stack?
5model/graph_attention_sparse/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5model/graph_attention_sparse/strided_slice_15/stack_1?
5model/graph_attention_sparse/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5model/graph_attention_sparse/strided_slice_15/stack_2?
-model/graph_attention_sparse/strided_slice_15StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0<model/graph_attention_sparse/strided_slice_15/stack:output:0>model/graph_attention_sparse/strided_slice_15/stack_1:output:0>model/graph_attention_sparse/strided_slice_15/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2/
-model/graph_attention_sparse/strided_slice_15?
-model/graph_attention_sparse/GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model/graph_attention_sparse/GatherV2_15/axis?
(model/graph_attention_sparse/GatherV2_15GatherV20model/graph_attention_sparse/Reshape_15:output:06model/graph_attention_sparse/strided_slice_15:output:06model/graph_attention_sparse/GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2*
(model/graph_attention_sparse/GatherV2_15?
"model/graph_attention_sparse/add_7AddV21model/graph_attention_sparse/GatherV2_14:output:01model/graph_attention_sparse/GatherV2_15:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse/add_7?
4model/graph_attention_sparse/leaky_re_lu_7/LeakyRelu	LeakyRelu&model/graph_attention_sparse/add_7:z:0*#
_output_shapes
:?????????26
4model/graph_attention_sparse/leaky_re_lu_7/LeakyRelu?
0model/graph_attention_sparse/dropout_14/IdentityIdentity0model/graph_attention_sparse/MatMul_21:product:0*
T0*
_output_shapes
:	?22
0model/graph_attention_sparse/dropout_14/Identity?
0model/graph_attention_sparse/dropout_15/IdentityIdentityBmodel/graph_attention_sparse/leaky_re_lu_7/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????22
0model/graph_attention_sparse/dropout_15/Identity?
7model/graph_attention_sparse/SparseTensor_7/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse/SparseTensor_7/dense_shape?
:model/graph_attention_sparse/SparseSoftmax_7/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:09model/graph_attention_sparse/dropout_15/Identity:output:0@model/graph_attention_sparse/SparseTensor_7/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse/SparseSoftmax_7/SparseSoftmax?
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse/SparseSoftmax_7/SparseSoftmax:output:0@model/graph_attention_sparse/SparseTensor_7/dense_shape:output:09model/graph_attention_sparse/dropout_14/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMul?
5model/graph_attention_sparse/BiasAdd_7/ReadVariableOpReadVariableOp>model_graph_attention_sparse_biasadd_7_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse/BiasAdd_7/ReadVariableOp?
&model/graph_attention_sparse/BiasAdd_7BiasAddXmodel/graph_attention_sparse/SparseTensorDenseMatMul_7/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse/BiasAdd_7?
(model/graph_attention_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(model/graph_attention_sparse/concat/axis?
#model/graph_attention_sparse/concatConcatV2-model/graph_attention_sparse/BiasAdd:output:0/model/graph_attention_sparse/BiasAdd_1:output:0/model/graph_attention_sparse/BiasAdd_2:output:0/model/graph_attention_sparse/BiasAdd_3:output:0/model/graph_attention_sparse/BiasAdd_4:output:0/model/graph_attention_sparse/BiasAdd_5:output:0/model/graph_attention_sparse/BiasAdd_6:output:0/model/graph_attention_sparse/BiasAdd_7:output:01model/graph_attention_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:	?@2%
#model/graph_attention_sparse/concat?
 model/graph_attention_sparse/EluElu,model/graph_attention_sparse/concat:output:0*
T0*
_output_shapes
:	?@2"
 model/graph_attention_sparse/Elu?
+model/graph_attention_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model/graph_attention_sparse/ExpandDims/dim?
'model/graph_attention_sparse/ExpandDims
ExpandDims.model/graph_attention_sparse/Elu:activations:04model/graph_attention_sparse/ExpandDims/dim:output:0*
T0*#
_output_shapes
:?@2)
'model/graph_attention_sparse/ExpandDims?
model/dropout_1/IdentityIdentity0model/graph_attention_sparse/ExpandDims:output:0*
T0*#
_output_shapes
:?@2
model/dropout_1/Identity?
&model/graph_attention_sparse_1/SqueezeSqueeze!model/dropout_1/Identity:output:0*
T0*
_output_shapes
:	?@*
squeeze_dims
 2(
&model/graph_attention_sparse_1/Squeeze?
4model/graph_attention_sparse_1/MatMul/ReadVariableOpReadVariableOp=model_graph_attention_sparse_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype026
4model/graph_attention_sparse_1/MatMul/ReadVariableOp?
%model/graph_attention_sparse_1/MatMulMatMul/model/graph_attention_sparse_1/Squeeze:output:0<model/graph_attention_sparse_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%model/graph_attention_sparse_1/MatMul?
6model/graph_attention_sparse_1/MatMul_1/ReadVariableOpReadVariableOp?model_graph_attention_sparse_1_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype028
6model/graph_attention_sparse_1/MatMul_1/ReadVariableOp?
'model/graph_attention_sparse_1/MatMul_1MatMul/model/graph_attention_sparse_1/MatMul:product:0>model/graph_attention_sparse_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2)
'model/graph_attention_sparse_1/MatMul_1?
6model/graph_attention_sparse_1/MatMul_2/ReadVariableOpReadVariableOp?model_graph_attention_sparse_1_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype028
6model/graph_attention_sparse_1/MatMul_2/ReadVariableOp?
'model/graph_attention_sparse_1/MatMul_2MatMul/model/graph_attention_sparse_1/MatMul:product:0>model/graph_attention_sparse_1/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2)
'model/graph_attention_sparse_1/MatMul_2?
,model/graph_attention_sparse_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,model/graph_attention_sparse_1/Reshape/shape?
&model/graph_attention_sparse_1/ReshapeReshape1model/graph_attention_sparse_1/MatMul_1:product:05model/graph_attention_sparse_1/Reshape/shape:output:0*
T0*
_output_shapes	
:?2(
&model/graph_attention_sparse_1/Reshape?
2model/graph_attention_sparse_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model/graph_attention_sparse_1/strided_slice/stack?
4model/graph_attention_sparse_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse_1/strided_slice/stack_1?
4model/graph_attention_sparse_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/graph_attention_sparse_1/strided_slice/stack_2?
,model/graph_attention_sparse_1/strided_sliceStridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse_1/strided_slice/stack:output:0=model/graph_attention_sparse_1/strided_slice/stack_1:output:0=model/graph_attention_sparse_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/graph_attention_sparse_1/strided_slice?
,model/graph_attention_sparse_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model/graph_attention_sparse_1/GatherV2/axis?
'model/graph_attention_sparse_1/GatherV2GatherV2/model/graph_attention_sparse_1/Reshape:output:05model/graph_attention_sparse_1/strided_slice:output:05model/graph_attention_sparse_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2)
'model/graph_attention_sparse_1/GatherV2?
.model/graph_attention_sparse_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????20
.model/graph_attention_sparse_1/Reshape_1/shape?
(model/graph_attention_sparse_1/Reshape_1Reshape1model/graph_attention_sparse_1/MatMul_2:product:07model/graph_attention_sparse_1/Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2*
(model/graph_attention_sparse_1/Reshape_1?
4model/graph_attention_sparse_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4model/graph_attention_sparse_1/strided_slice_1/stack?
6model/graph_attention_sparse_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6model/graph_attention_sparse_1/strided_slice_1/stack_1?
6model/graph_attention_sparse_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model/graph_attention_sparse_1/strided_slice_1/stack_2?
.model/graph_attention_sparse_1/strided_slice_1StridedSlice1model/squeezed_sparse_conversion/Squeeze:output:0=model/graph_attention_sparse_1/strided_slice_1/stack:output:0?model/graph_attention_sparse_1/strided_slice_1/stack_1:output:0?model/graph_attention_sparse_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.model/graph_attention_sparse_1/strided_slice_1?
.model/graph_attention_sparse_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.model/graph_attention_sparse_1/GatherV2_1/axis?
)model/graph_attention_sparse_1/GatherV2_1GatherV21model/graph_attention_sparse_1/Reshape_1:output:07model/graph_attention_sparse_1/strided_slice_1:output:07model/graph_attention_sparse_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2+
)model/graph_attention_sparse_1/GatherV2_1?
"model/graph_attention_sparse_1/addAddV20model/graph_attention_sparse_1/GatherV2:output:02model/graph_attention_sparse_1/GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2$
"model/graph_attention_sparse_1/add?
6model/graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu	LeakyRelu&model/graph_attention_sparse_1/add:z:0*#
_output_shapes
:?????????28
6model/graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu?
2model/graph_attention_sparse_1/dropout_16/IdentityIdentity/model/graph_attention_sparse_1/MatMul:product:0*
T0*
_output_shapes
:	?24
2model/graph_attention_sparse_1/dropout_16/Identity?
2model/graph_attention_sparse_1/dropout_17/IdentityIdentityDmodel/graph_attention_sparse_1/leaky_re_lu_8/LeakyRelu:activations:0*
T0*#
_output_shapes
:?????????24
2model/graph_attention_sparse_1/dropout_17/Identity?
7model/graph_attention_sparse_1/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      29
7model/graph_attention_sparse_1/SparseTensor/dense_shape?
:model/graph_attention_sparse_1/SparseSoftmax/SparseSoftmaxSparseSoftmax1model/squeezed_sparse_conversion/Squeeze:output:0;model/graph_attention_sparse_1/dropout_17/Identity:output:0@model/graph_attention_sparse_1/SparseTensor/dense_shape:output:0*
T0*#
_output_shapes
:?????????2<
:model/graph_attention_sparse_1/SparseSoftmax/SparseSoftmax?
Nmodel/graph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:0Cmodel/graph_attention_sparse_1/SparseSoftmax/SparseSoftmax:output:0@model/graph_attention_sparse_1/SparseTensor/dense_shape:output:0;model/graph_attention_sparse_1/dropout_16/Identity:output:0*
T0*
_output_shapes
:	?2P
Nmodel/graph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
5model/graph_attention_sparse_1/BiasAdd/ReadVariableOpReadVariableOp>model_graph_attention_sparse_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5model/graph_attention_sparse_1/BiasAdd/ReadVariableOp?
&model/graph_attention_sparse_1/BiasAddBiasAddXmodel/graph_attention_sparse_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0=model/graph_attention_sparse_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse_1/BiasAdd?
$model/graph_attention_sparse_1/stackPack/model/graph_attention_sparse_1/BiasAdd:output:0*
N*
T0*#
_output_shapes
:?2&
$model/graph_attention_sparse_1/stack?
5model/graph_attention_sparse_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 27
5model/graph_attention_sparse_1/Mean/reduction_indices?
#model/graph_attention_sparse_1/MeanMean-model/graph_attention_sparse_1/stack:output:0>model/graph_attention_sparse_1/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	?2%
#model/graph_attention_sparse_1/Mean?
&model/graph_attention_sparse_1/SoftmaxSoftmax,model/graph_attention_sparse_1/Mean:output:0*
T0*
_output_shapes
:	?2(
&model/graph_attention_sparse_1/Softmax?
-model/graph_attention_sparse_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model/graph_attention_sparse_1/ExpandDims/dim?
)model/graph_attention_sparse_1/ExpandDims
ExpandDims0model/graph_attention_sparse_1/Softmax:softmax:06model/graph_attention_sparse_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:?2+
)model/graph_attention_sparse_1/ExpandDims?
"model/gather_indices/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/gather_indices/GatherV2/axis?
model/gather_indices/GatherV2GatherV22model/graph_attention_sparse_1/ExpandDims:output:0input_2+model/gather_indices/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????*

batch_dims2
model/gather_indices/GatherV2?
IdentityIdentity&model/gather_indices/GatherV2:output:04^model/graph_attention_sparse/BiasAdd/ReadVariableOp6^model/graph_attention_sparse/BiasAdd_1/ReadVariableOp6^model/graph_attention_sparse/BiasAdd_2/ReadVariableOp6^model/graph_attention_sparse/BiasAdd_3/ReadVariableOp6^model/graph_attention_sparse/BiasAdd_4/ReadVariableOp6^model/graph_attention_sparse/BiasAdd_5/ReadVariableOp6^model/graph_attention_sparse/BiasAdd_6/ReadVariableOp6^model/graph_attention_sparse/BiasAdd_7/ReadVariableOp3^model/graph_attention_sparse/MatMul/ReadVariableOp5^model/graph_attention_sparse/MatMul_1/ReadVariableOp6^model/graph_attention_sparse/MatMul_10/ReadVariableOp6^model/graph_attention_sparse/MatMul_11/ReadVariableOp6^model/graph_attention_sparse/MatMul_12/ReadVariableOp6^model/graph_attention_sparse/MatMul_13/ReadVariableOp6^model/graph_attention_sparse/MatMul_14/ReadVariableOp6^model/graph_attention_sparse/MatMul_15/ReadVariableOp6^model/graph_attention_sparse/MatMul_16/ReadVariableOp6^model/graph_attention_sparse/MatMul_17/ReadVariableOp6^model/graph_attention_sparse/MatMul_18/ReadVariableOp6^model/graph_attention_sparse/MatMul_19/ReadVariableOp5^model/graph_attention_sparse/MatMul_2/ReadVariableOp6^model/graph_attention_sparse/MatMul_20/ReadVariableOp6^model/graph_attention_sparse/MatMul_21/ReadVariableOp6^model/graph_attention_sparse/MatMul_22/ReadVariableOp6^model/graph_attention_sparse/MatMul_23/ReadVariableOp5^model/graph_attention_sparse/MatMul_3/ReadVariableOp5^model/graph_attention_sparse/MatMul_4/ReadVariableOp5^model/graph_attention_sparse/MatMul_5/ReadVariableOp5^model/graph_attention_sparse/MatMul_6/ReadVariableOp5^model/graph_attention_sparse/MatMul_7/ReadVariableOp5^model/graph_attention_sparse/MatMul_8/ReadVariableOp5^model/graph_attention_sparse/MatMul_9/ReadVariableOp6^model/graph_attention_sparse_1/BiasAdd/ReadVariableOp5^model/graph_attention_sparse_1/MatMul/ReadVariableOp7^model/graph_attention_sparse_1/MatMul_1/ReadVariableOp7^model/graph_attention_sparse_1/MatMul_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::2j
3model/graph_attention_sparse/BiasAdd/ReadVariableOp3model/graph_attention_sparse/BiasAdd/ReadVariableOp2n
5model/graph_attention_sparse/BiasAdd_1/ReadVariableOp5model/graph_attention_sparse/BiasAdd_1/ReadVariableOp2n
5model/graph_attention_sparse/BiasAdd_2/ReadVariableOp5model/graph_attention_sparse/BiasAdd_2/ReadVariableOp2n
5model/graph_attention_sparse/BiasAdd_3/ReadVariableOp5model/graph_attention_sparse/BiasAdd_3/ReadVariableOp2n
5model/graph_attention_sparse/BiasAdd_4/ReadVariableOp5model/graph_attention_sparse/BiasAdd_4/ReadVariableOp2n
5model/graph_attention_sparse/BiasAdd_5/ReadVariableOp5model/graph_attention_sparse/BiasAdd_5/ReadVariableOp2n
5model/graph_attention_sparse/BiasAdd_6/ReadVariableOp5model/graph_attention_sparse/BiasAdd_6/ReadVariableOp2n
5model/graph_attention_sparse/BiasAdd_7/ReadVariableOp5model/graph_attention_sparse/BiasAdd_7/ReadVariableOp2h
2model/graph_attention_sparse/MatMul/ReadVariableOp2model/graph_attention_sparse/MatMul/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_1/ReadVariableOp4model/graph_attention_sparse/MatMul_1/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_10/ReadVariableOp5model/graph_attention_sparse/MatMul_10/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_11/ReadVariableOp5model/graph_attention_sparse/MatMul_11/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_12/ReadVariableOp5model/graph_attention_sparse/MatMul_12/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_13/ReadVariableOp5model/graph_attention_sparse/MatMul_13/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_14/ReadVariableOp5model/graph_attention_sparse/MatMul_14/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_15/ReadVariableOp5model/graph_attention_sparse/MatMul_15/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_16/ReadVariableOp5model/graph_attention_sparse/MatMul_16/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_17/ReadVariableOp5model/graph_attention_sparse/MatMul_17/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_18/ReadVariableOp5model/graph_attention_sparse/MatMul_18/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_19/ReadVariableOp5model/graph_attention_sparse/MatMul_19/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_2/ReadVariableOp4model/graph_attention_sparse/MatMul_2/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_20/ReadVariableOp5model/graph_attention_sparse/MatMul_20/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_21/ReadVariableOp5model/graph_attention_sparse/MatMul_21/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_22/ReadVariableOp5model/graph_attention_sparse/MatMul_22/ReadVariableOp2n
5model/graph_attention_sparse/MatMul_23/ReadVariableOp5model/graph_attention_sparse/MatMul_23/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_3/ReadVariableOp4model/graph_attention_sparse/MatMul_3/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_4/ReadVariableOp4model/graph_attention_sparse/MatMul_4/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_5/ReadVariableOp4model/graph_attention_sparse/MatMul_5/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_6/ReadVariableOp4model/graph_attention_sparse/MatMul_6/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_7/ReadVariableOp4model/graph_attention_sparse/MatMul_7/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_8/ReadVariableOp4model/graph_attention_sparse/MatMul_8/ReadVariableOp2l
4model/graph_attention_sparse/MatMul_9/ReadVariableOp4model/graph_attention_sparse/MatMul_9/ReadVariableOp2n
5model/graph_attention_sparse_1/BiasAdd/ReadVariableOp5model/graph_attention_sparse_1/BiasAdd/ReadVariableOp2l
4model/graph_attention_sparse_1/MatMul/ReadVariableOp4model/graph_attention_sparse_1/MatMul/ReadVariableOp2p
6model/graph_attention_sparse_1/MatMul_1/ReadVariableOp6model/graph_attention_sparse_1/MatMul_1/ReadVariableOp2p
6model/graph_attention_sparse_1/MatMul_2/ReadVariableOp6model/graph_attention_sparse_1/MatMul_2/ReadVariableOp:M I
$
_output_shapes
:??
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?
C
'__inference_dropout_layer_call_fn_11140

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_84692
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:??2

Identity"
identityIdentity:output:0*#
_input_shapes
:??:L H
$
_output_shapes
:??
 
_user_specified_nameinputs
?
Z
.__inference_gather_indices_layer_call_fn_12186
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gather_indices_layer_call_and_return_conditional_losses_95832
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*5
_input_shapes$
":?:?????????:M I
#
_output_shapes
:?
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
b
)__inference_dropout_1_layer_call_fn_12024

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_94032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?
r
H__inference_gather_indices_layer_call_and_return_conditional_losses_9583

inputs
inputs_1
identity`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2inputsinputs_1GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????*

batch_dims2

GatherV2i
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*5
_input_shapes$
":?:?????????:K G
#
_output_shapes
:?
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
R__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_9484

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource$
 matmul_2_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOpf
SqueezeSqueezeinputs*
T0*
_output_shapes
:	?@*
squeeze_dims
 2	
Squeeze?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulSqueeze:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulMatMul:product:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp{
MatMul_2MatMulMatMul:product:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2

MatMul_2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeo
ReshapeReshapeMatMul_1:product:0Reshape/shape:output:0*
T0*
_output_shapes	
:?2	
Reshape{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape:output:0strided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2u
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeu
	Reshape_1ReshapeMatMul_2:product:0Reshape_1/shape:output:0*
T0*
_output_shapes	
:?2
	Reshape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_1:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1i
addAddV2GatherV2:output:0GatherV2_1:output:0*
T0*#
_output_shapes
:?????????2
addi
leaky_re_lu/LeakyRelu	LeakyReluadd:z:0*#
_output_shapes
:?????????2
leaky_re_lu/LeakyRelus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulMatMul:product:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?
     2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	?2
dropout/dropout/Mul_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul#leaky_re_lu/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
SparseTensor_2/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"?
      ?
      2
SparseTensor_2/dense_shape?
SparseSoftmax/SparseSoftmaxSparseSoftmaxinputs_1dropout_1/dropout/Mul_1:z:0#SparseTensor_2/dense_shape:output:0*
T0*#
_output_shapes
:?????????2
SparseSoftmax/SparseSoftmax?
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1$SparseSoftmax/SparseSoftmax:output:0#SparseTensor_2/dense_shape:output:0dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:	?21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAdd_
stackPackBiasAdd:output:0*
N*
T0*#
_output_shapes
:?2
stackr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indiceso
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:	?2
MeanV
SoftmaxSoftmaxMean:output:0*
T0*
_output_shapes
:	?2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:?2

ExpandDims?
IdentityIdentityExpandDims:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp*
T0*#
_output_shapes
:?2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?@:?????????:?????????:::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
?
6__inference_graph_attention_sparse_layer_call_fn_11930
inputs_0

inputs	
inputs_1
inputs_2	
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*/
Tin(
&2$		*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@*B
_read_only_resource_inputs$
" 	
 !"#*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_88872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
$
_output_shapes
:??
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_9871
input_1
input_2
input_3	
input_4
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_97962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
$
_output_shapes
:??
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_11125

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constp
dropout/MulMulinputsdropout/Const:output:0*
T0*$
_output_shapes
:??2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?
  ?  2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:??*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:??2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:??2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:??2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:??2

Identity"
identityIdentity:output:0*#
_input_shapes
:??:L H
$
_output_shapes
:??
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
8
input_1-
serving_default_input_1:0??
;
input_20
serving_default_input_2:0?????????
?
input_34
serving_default_input_3:0	?????????
;
input_40
serving_default_input_4:0?????????>
lambda4
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?T
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer-9
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?Q
_tf_keras_network?Q{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 2708, 1440]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null, 2]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "SqueezedSparseConversion", "config": {"shape": {"class_name": "__tuple__", "items": [2708, 2708]}, "dtype": "float32"}, "name": "squeezed_sparse_conversion", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "GraphAttentionSparse", "config": {"name": "graph_attention_sparse", "trainable": true, "dtype": "float32", "units": 8, "attn_heads": 8, "attn_heads_reduction": "concat", "in_dropout_rate": 0.5, "attn_dropout_rate": 0.5, "activation": "elu", "use_bias": true, "saliency_map_support": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "attn_kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "attn_kernel_regularizer": null, "attn_kernel_constraint": null}, "name": "graph_attention_sparse", "inbound_nodes": [[["dropout", 0, 0, {}], ["squeezed_sparse_conversion", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["graph_attention_sparse", 0, 0, {}]]]}, {"class_name": "GraphAttentionSparse", "config": {"name": "graph_attention_sparse_1", "trainable": true, "dtype": "float32", "units": 7, "attn_heads": 1, "attn_heads_reduction": "average", "in_dropout_rate": 0.5, "attn_dropout_rate": 0.5, "activation": "softmax", "use_bias": true, "saliency_map_support": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "attn_kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "attn_kernel_regularizer": null, "attn_kernel_constraint": null}, "name": "graph_attention_sparse_1", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["squeezed_sparse_conversion", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "GatherIndices", "config": {"name": "gather_indices", "trainable": true, "dtype": "float32", "axis": null, "batch_dims": 1}, "name": "gather_indices", "inbound_nodes": [[["graph_attention_sparse_1", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAABAAAAUwAAAHMEAAAAfABTACkBTqkAqQHaAXhyAQAAAHIBAAAA+m0v\nVXNlcnMvamtpbS9DbGFzc2VzL0RUTS9Qcm9qZWN0L1B5dGhvbi92ZW52L2xpYi9weXRob24zLjgv\nc2l0ZS1wYWNrYWdlcy9zdGVsbGFyZ3JhcGgvbGF5ZXIvZ3JhcGhfYXR0ZW50aW9uLnB52gg8bGFt\nYmRhPhMDAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.graph_attention", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["gather_indices", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2708, 1440]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [1, 2708, 1440]}, {"class_name": "TensorShape", "items": [1, null]}, {"class_name": "TensorShape", "items": [1, null, 2]}, {"class_name": "TensorShape", "items": [1, null]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 2708, 1440]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null, 2]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "SqueezedSparseConversion", "config": {"shape": {"class_name": "__tuple__", "items": [2708, 2708]}, "dtype": "float32"}, "name": "squeezed_sparse_conversion", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "GraphAttentionSparse", "config": {"name": "graph_attention_sparse", "trainable": true, "dtype": "float32", "units": 8, "attn_heads": 8, "attn_heads_reduction": "concat", "in_dropout_rate": 0.5, "attn_dropout_rate": 0.5, "activation": "elu", "use_bias": true, "saliency_map_support": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "attn_kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "attn_kernel_regularizer": null, "attn_kernel_constraint": null}, "name": "graph_attention_sparse", "inbound_nodes": [[["dropout", 0, 0, {}], ["squeezed_sparse_conversion", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["graph_attention_sparse", 0, 0, {}]]]}, {"class_name": "GraphAttentionSparse", "config": {"name": "graph_attention_sparse_1", "trainable": true, "dtype": "float32", "units": 7, "attn_heads": 1, "attn_heads_reduction": "average", "in_dropout_rate": 0.5, "attn_dropout_rate": 0.5, "activation": "softmax", "use_bias": true, "saliency_map_support": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "attn_kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "attn_kernel_regularizer": null, "attn_kernel_constraint": null}, "name": "graph_attention_sparse_1", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["squeezed_sparse_conversion", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "GatherIndices", "config": {"name": "gather_indices", "trainable": true, "dtype": "float32", "axis": null, "batch_dims": 1}, "name": "gather_indices", "inbound_nodes": [[["graph_attention_sparse_1", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAABAAAAUwAAAHMEAAAAfABTACkBTqkAqQHaAXhyAQAAAHIBAAAA+m0v\nVXNlcnMvamtpbS9DbGFzc2VzL0RUTS9Qcm9qZWN0L1B5dGhvbi92ZW52L2xpYi9weXRob24zLjgv\nc2l0ZS1wYWNrYWdlcy9zdGVsbGFyZ3JhcGgvbGF5ZXIvZ3JhcGhfYXR0ZW50aW9uLnB52gg8bGFt\nYmRhPhMDAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.graph_attention", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["gather_indices", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["lambda", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.004999999888241291, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 2708, 1440]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 2708, 1440]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "int64", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null, 2]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "input_3"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SqueezedSparseConversion", "name": "squeezed_sparse_conversion", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"shape": {"class_name": "__tuple__", "items": [2708, 2708]}, "dtype": "float32"}}
?
kernels

biases
attn_kernels
ig_delta
	delta
ig_non_exist_edge
non_exist_edge
kernel_0

 bias_0
!attn_kernel_self_0
"attn_kernel_neigh_0
#kernel_1

$bias_1
%attn_kernel_self_1
&attn_kernel_neigh_1
'kernel_2

(bias_2
)attn_kernel_self_2
*attn_kernel_neigh_2
+kernel_3

,bias_3
-attn_kernel_self_3
.attn_kernel_neigh_3
/kernel_4

0bias_4
1attn_kernel_self_4
2attn_kernel_neigh_4
3kernel_5

4bias_5
5attn_kernel_self_5
6attn_kernel_neigh_5
7kernel_6

8bias_6
9attn_kernel_self_6
:attn_kernel_neigh_6
;kernel_7

<bias_7
=attn_kernel_self_7
>attn_kernel_neigh_7
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GraphAttentionSparse", "name": "graph_attention_sparse", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "graph_attention_sparse", "trainable": true, "dtype": "float32", "units": 8, "attn_heads": 8, "attn_heads_reduction": "concat", "in_dropout_rate": 0.5, "attn_dropout_rate": 0.5, "activation": "elu", "use_bias": true, "saliency_map_support": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "attn_kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "attn_kernel_regularizer": null, "attn_kernel_constraint": null}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 2708, 1440]}, {"class_name": "TensorShape", "items": [2708, 2708]}]}
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Gkernels

Hbiases
Iattn_kernels
Jig_delta
	Jdelta
Kig_non_exist_edge
Knon_exist_edge
Lkernel_0

Mbias_0
Nattn_kernel_self_0
Oattn_kernel_neigh_0
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GraphAttentionSparse", "name": "graph_attention_sparse_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "graph_attention_sparse_1", "trainable": true, "dtype": "float32", "units": 7, "attn_heads": 1, "attn_heads_reduction": "average", "in_dropout_rate": 0.5, "attn_dropout_rate": 0.5, "activation": "softmax", "use_bias": true, "saliency_map_support": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "attn_kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "attn_kernel_regularizer": null, "attn_kernel_constraint": null}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 2708, 64]}, {"class_name": "TensorShape", "items": [2708, 2708]}]}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_2"}}
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GatherIndices", "name": "gather_indices", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gather_indices", "trainable": true, "dtype": "float32", "axis": null, "batch_dims": 1}}
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAABAAAAUwAAAHMEAAAAfABTACkBTqkAqQHaAXhyAQAAAHIBAAAA+m0v\nVXNlcnMvamtpbS9DbGFzc2VzL0RUTS9Qcm9qZWN0L1B5dGhvbi92ZW52L2xpYi9weXRob24zLjgv\nc2l0ZS1wYWNrYWdlcy9zdGVsbGFyZ3JhcGgvbGF5ZXIvZ3JhcGhfYXR0ZW50aW9uLnB52gg8bGFt\nYmRhPhMDAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.graph_attention", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
\iter

]beta_1

^beta_2
	_decay
`learning_ratem? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m?Lm?Mm?Nm?Om?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v?Lv?Mv?Nv?Ov?"
	optimizer
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31
32
33
L34
M35
N36
O37
J38
K39"
trackable_list_wrapper
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31
L32
M33
N34
O35"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses
alayer_metrics
bmetrics
cnon_trainable_variables
dlayer_regularization_losses

elayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses
flayer_metrics
gmetrics
hnon_trainable_variables
ilayer_regularization_losses

jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses
klayer_metrics
lmetrics
mnon_trainable_variables
nlayer_regularization_losses

olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
0
#1
'2
+3
/4
35
76
;7"
trackable_list_wrapper
X
 0
$1
(2
,3
04
45
86
<7"
trackable_list_wrapper
X
p0
q1
r2
s3
t4
u5
v6
w7"
trackable_list_wrapper
':% 2graph_attention_sparse/ig_delta
0:. 2(graph_attention_sparse/ig_non_exist_edge
2:0	?2graph_attention_sparse/kernel_0
+:)2graph_attention_sparse/bias_0
;:92)graph_attention_sparse/attn_kernel_self_0
<::2*graph_attention_sparse/attn_kernel_neigh_0
2:0	?2graph_attention_sparse/kernel_1
+:)2graph_attention_sparse/bias_1
;:92)graph_attention_sparse/attn_kernel_self_1
<::2*graph_attention_sparse/attn_kernel_neigh_1
2:0	?2graph_attention_sparse/kernel_2
+:)2graph_attention_sparse/bias_2
;:92)graph_attention_sparse/attn_kernel_self_2
<::2*graph_attention_sparse/attn_kernel_neigh_2
2:0	?2graph_attention_sparse/kernel_3
+:)2graph_attention_sparse/bias_3
;:92)graph_attention_sparse/attn_kernel_self_3
<::2*graph_attention_sparse/attn_kernel_neigh_3
2:0	?2graph_attention_sparse/kernel_4
+:)2graph_attention_sparse/bias_4
;:92)graph_attention_sparse/attn_kernel_self_4
<::2*graph_attention_sparse/attn_kernel_neigh_4
2:0	?2graph_attention_sparse/kernel_5
+:)2graph_attention_sparse/bias_5
;:92)graph_attention_sparse/attn_kernel_self_5
<::2*graph_attention_sparse/attn_kernel_neigh_5
2:0	?2graph_attention_sparse/kernel_6
+:)2graph_attention_sparse/bias_6
;:92)graph_attention_sparse/attn_kernel_self_6
<::2*graph_attention_sparse/attn_kernel_neigh_6
2:0	?2graph_attention_sparse/kernel_7
+:)2graph_attention_sparse/bias_7
;:92)graph_attention_sparse/attn_kernel_self_7
<::2*graph_attention_sparse/attn_kernel_neigh_7
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31
32
33"
trackable_list_wrapper
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
724
825
926
:27
;28
<29
=30
>31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
@trainable_variables
Aregularization_losses
xlayer_metrics
ymetrics
znon_trainable_variables
{layer_regularization_losses

|layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
C	variables
Dtrainable_variables
Eregularization_losses
}layer_metrics
~metrics
non_trainable_variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
L0"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
):' 2!graph_attention_sparse_1/ig_delta
2:0 2*graph_attention_sparse_1/ig_non_exist_edge
3:1@2!graph_attention_sparse_1/kernel_0
-:+2graph_attention_sparse_1/bias_0
=:;2+graph_attention_sparse_1/attn_kernel_self_0
>:<2,graph_attention_sparse_1/attn_kernel_neigh_0
J
L0
M1
N2
O3
J4
K5"
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
P	variables
Qtrainable_variables
Rregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
T	variables
Utrainable_variables
Vregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
X	variables
Ytrainable_variables
Zregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
<
0
1
J2
K3"
trackable_list_wrapper
 "
trackable_list_wrapper
n
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
10"
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
.
!0
"1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
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
.
N0
O1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
J0
K1"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
7:5	?2&Adam/graph_attention_sparse/kernel_0/m
0:.2$Adam/graph_attention_sparse/bias_0/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_0/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_0/m
7:5	?2&Adam/graph_attention_sparse/kernel_1/m
0:.2$Adam/graph_attention_sparse/bias_1/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_1/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_1/m
7:5	?2&Adam/graph_attention_sparse/kernel_2/m
0:.2$Adam/graph_attention_sparse/bias_2/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_2/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_2/m
7:5	?2&Adam/graph_attention_sparse/kernel_3/m
0:.2$Adam/graph_attention_sparse/bias_3/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_3/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_3/m
7:5	?2&Adam/graph_attention_sparse/kernel_4/m
0:.2$Adam/graph_attention_sparse/bias_4/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_4/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_4/m
7:5	?2&Adam/graph_attention_sparse/kernel_5/m
0:.2$Adam/graph_attention_sparse/bias_5/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_5/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_5/m
7:5	?2&Adam/graph_attention_sparse/kernel_6/m
0:.2$Adam/graph_attention_sparse/bias_6/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_6/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_6/m
7:5	?2&Adam/graph_attention_sparse/kernel_7/m
0:.2$Adam/graph_attention_sparse/bias_7/m
@:>20Adam/graph_attention_sparse/attn_kernel_self_7/m
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_7/m
8:6@2(Adam/graph_attention_sparse_1/kernel_0/m
2:02&Adam/graph_attention_sparse_1/bias_0/m
B:@22Adam/graph_attention_sparse_1/attn_kernel_self_0/m
C:A23Adam/graph_attention_sparse_1/attn_kernel_neigh_0/m
7:5	?2&Adam/graph_attention_sparse/kernel_0/v
0:.2$Adam/graph_attention_sparse/bias_0/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_0/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_0/v
7:5	?2&Adam/graph_attention_sparse/kernel_1/v
0:.2$Adam/graph_attention_sparse/bias_1/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_1/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_1/v
7:5	?2&Adam/graph_attention_sparse/kernel_2/v
0:.2$Adam/graph_attention_sparse/bias_2/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_2/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_2/v
7:5	?2&Adam/graph_attention_sparse/kernel_3/v
0:.2$Adam/graph_attention_sparse/bias_3/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_3/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_3/v
7:5	?2&Adam/graph_attention_sparse/kernel_4/v
0:.2$Adam/graph_attention_sparse/bias_4/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_4/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_4/v
7:5	?2&Adam/graph_attention_sparse/kernel_5/v
0:.2$Adam/graph_attention_sparse/bias_5/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_5/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_5/v
7:5	?2&Adam/graph_attention_sparse/kernel_6/v
0:.2$Adam/graph_attention_sparse/bias_6/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_6/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_6/v
7:5	?2&Adam/graph_attention_sparse/kernel_7/v
0:.2$Adam/graph_attention_sparse/bias_7/v
@:>20Adam/graph_attention_sparse/attn_kernel_self_7/v
A:?21Adam/graph_attention_sparse/attn_kernel_neigh_7/v
8:6@2(Adam/graph_attention_sparse_1/kernel_0/v
2:02&Adam/graph_attention_sparse_1/bias_0/v
B:@22Adam/graph_attention_sparse_1/attn_kernel_self_0/v
C:A23Adam/graph_attention_sparse_1/attn_kernel_neigh_0/v
?2?
__inference__wrapped_model_8420?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
?
input_1??
!?
input_2?????????
%?"
input_3?????????	
!?
input_4?????????
?2?
@__inference_model_layer_call_and_return_conditional_losses_10611
@__inference_model_layer_call_and_return_conditional_losses_10953
?__inference_model_layer_call_and_return_conditional_losses_9702
?__inference_model_layer_call_and_return_conditional_losses_9614?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_9871
%__inference_model_layer_call_fn_10039
%__inference_model_layer_call_fn_11033
%__inference_model_layer_call_fn_11113?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_11140
'__inference_dropout_layer_call_fn_11135?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_11125
B__inference_dropout_layer_call_and_return_conditional_losses_11130?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
:__inference_squeezed_sparse_conversion_layer_call_fn_11160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_11150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_graph_attention_sparse_layer_call_fn_11930
6__inference_graph_attention_sparse_layer_call_fn_12002?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
Q__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_11565
Q__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_11858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_12024
)__inference_dropout_1_layer_call_fn_12029?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_12019
D__inference_dropout_1_layer_call_and_return_conditional_losses_12014?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_graph_attention_sparse_1_layer_call_fn_12173
8__inference_graph_attention_sparse_1_layer_call_fn_12157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
S__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_12092
S__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_12141?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
.__inference_gather_indices_layer_call_fn_12186?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_gather_indices_layer_call_and_return_conditional_losses_12180?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_lambda_layer_call_fn_12199
&__inference_lambda_layer_call_fn_12204?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_lambda_layer_call_and_return_conditional_losses_12194
A__inference_lambda_layer_call_and_return_conditional_losses_12190?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_10129input_1input_2input_3input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_8420?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
input_1??
!?
input_2?????????
%?"
input_3?????????	
!?
input_4?????????
? "3?0
.
lambda$?!
lambda??????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_12014T/?,
%?"
?
inputs?@
p
? "!?
?
0?@
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_12019T/?,
%?"
?
inputs?@
p 
? "!?
?
0?@
? t
)__inference_dropout_1_layer_call_fn_12024G/?,
%?"
?
inputs?@
p
? "??@t
)__inference_dropout_1_layer_call_fn_12029G/?,
%?"
?
inputs?@
p 
? "??@?
B__inference_dropout_layer_call_and_return_conditional_losses_11125V0?-
&?#
?
inputs??
p
? ""?
?
0??
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_11130V0?-
&?#
?
inputs??
p 
? ""?
?
0??
? t
'__inference_dropout_layer_call_fn_11135I0?-
&?#
?
inputs??
p
? "???t
'__inference_dropout_layer_call_fn_11140I0?-
&?#
?
inputs??
p 
? "????
I__inference_gather_indices_layer_call_and_return_conditional_losses_12180?V?S
L?I
G?D
?
inputs/0?
"?
inputs/1?????????
? ")?&
?
0?????????
? ?
.__inference_gather_indices_layer_call_fn_12186vV?S
L?I
G?D
?
inputs/0?
"?
inputs/1?????????
? "???????????
S__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_12092?LNOM???
j?g
e?b
?
inputs/0?@
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp"!?
?
0?
? ?
S__inference_graph_attention_sparse_1_layer_call_and_return_conditional_losses_12141?LNOM???
j?g
e?b
?
inputs/0?@
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp "!?
?
0?
? ?
8__inference_graph_attention_sparse_1_layer_call_fn_12157?LNOM???
j?g
e?b
?
inputs/0?@
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp"???
8__inference_graph_attention_sparse_1_layer_call_fn_12173?LNOM???
j?g
e?b
?
inputs/0?@
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp "???
Q__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_11565? !" #%&$')*(+-.,/120356479:8;=><???
k?h
f?c
?
inputs/0??
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp"!?
?
0?@
? ?
Q__inference_graph_attention_sparse_layer_call_and_return_conditional_losses_11858? !" #%&$')*(+-.,/120356479:8;=><???
k?h
f?c
?
inputs/0??
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp "!?
?
0?@
? ?
6__inference_graph_attention_sparse_layer_call_fn_11930? !" #%&$')*(+-.,/120356479:8;=><???
k?h
f?c
?
inputs/0??
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp"??@?
6__inference_graph_attention_sparse_layer_call_fn_12002? !" #%&$')*(+-.,/120356479:8;=><???
k?h
f?c
?
inputs/0??
@?='?$
???????????????????
?SparseTensorSpec
?

trainingp "??@?
A__inference_lambda_layer_call_and_return_conditional_losses_12190h;?8
1?.
$?!
inputs?????????

 
p
? ")?&
?
0?????????
? ?
A__inference_lambda_layer_call_and_return_conditional_losses_12194h;?8
1?.
$?!
inputs?????????

 
p 
? ")?&
?
0?????????
? ?
&__inference_lambda_layer_call_fn_12199[;?8
1?.
$?!
inputs?????????

 
p
? "???????????
&__inference_lambda_layer_call_fn_12204[;?8
1?.
$?!
inputs?????????

 
p 
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_10611?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
inputs/0??
"?
inputs/1?????????
&?#
inputs/2?????????	
"?
inputs/3?????????
p

 
? ")?&
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_10953?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
inputs/0??
"?
inputs/1?????????
&?#
inputs/2?????????	
"?
inputs/3?????????
p 

 
? ")?&
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_9614?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
input_1??
!?
input_2?????????
%?"
input_3?????????	
!?
input_4?????????
p

 
? ")?&
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_9702?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
input_1??
!?
input_2?????????
%?"
input_3?????????	
!?
input_4?????????
p 

 
? ")?&
?
0?????????
? ?
%__inference_model_layer_call_fn_10039?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
input_1??
!?
input_2?????????
%?"
input_3?????????	
!?
input_4?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_11033?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
inputs/0??
"?
inputs/1?????????
&?#
inputs/2?????????	
"?
inputs/3?????????
p

 
? "???????????
%__inference_model_layer_call_fn_11113?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
inputs/0??
"?
inputs/1?????????
&?#
inputs/2?????????	
"?
inputs/3?????????
p 

 
? "???????????
$__inference_model_layer_call_fn_9871?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
???
???
?
input_1??
!?
input_2?????????
%?"
input_3?????????	
!?
input_4?????????
p

 
? "???????????
#__inference_signature_wrapper_10129?$!" #%&$')*(+-.,/120356479:8;=><LNOM???
? 
???
)
input_1?
input_1??
,
input_2!?
input_2?????????
0
input_3%?"
input_3?????????	
,
input_4!?
input_4?????????"3?0
.
lambda$?!
lambda??????????
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_11150?^?[
T?Q
O?L
&?#
inputs/0?????????	
"?
inputs/1?????????
? ":?7
0?-?
?
??
?SparseTensorSpec
? ?
:__inference_squeezed_sparse_conversion_layer_call_fn_11160?^?[
T?Q
O?L
&?#
inputs/0?????????	
"?
inputs/1?????????
? "@?='?$
???????????????????
?SparseTensorSpec