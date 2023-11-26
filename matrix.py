import copy
import numpy as np

class Matrix:
    element = []
    shape = []
    def __init__(self,value,shape=None):
        elm = []
        if isinstance(value[0],list): # 若用户输入二维数组，需要转化为一维数组
            for i in value:
                elm += i
            n = len(value) # 行数
            m = len(value[0]) # 列数
        elif isinstance(value,np.ndarray):
            elm = [i for i in value.flatten()]
            n = value.shape[0]
            m = 1
            if len(value.shape)>1:
                m = value.shape[1]
        elif isinstance(value[0],(int,float,np.number,complex)): # 若用户输入一维数组，直接拷贝
            elm = value
            n = len(value) # 默认输入为列向量
            m = 1
        else:
            raise NotImplementedError(f'Unknown Parameters!{value}')

        self.element = elm # 本类矩阵元素都是存储为一维数组
        if shape: # 若用户未定义形状，需要自行计算形状
            n, m = shape[0], shape[1]
            if n*m!=len(self.element):
                raise ValueError(f'Shape invalid! User defined shape:{n,m} not match element number:[{len(self.element)}].')
        self.shape = [n,m]

    def __str__(self):
        n,m = self.shape
        ret = '[\n'
        for i in range(0,n):

            ret +='\t['
            for j in range(0,m):
                t = self.element[i*m+j]
                if isClose(t): # 优化零的显示
                    t = 0
                ret += f'{t:.2e} '
            ret +=']\n'
        ret +=f'],Shape:[{n},{m}]'
        return ret
    
    def __add__(self,B):
        if self.shape != B.shape:
            raise ValueError(f'Shape not match! Can not add two matrix with shape {self.shape} and {B.shape}')
        ret_value = [self.element[i]+B.element[i] for i in range(0,len(self.element))]
        return Matrix(ret_value,self.shape)
    
    def __sub__(self,B):
        if self.shape != B.shape:
            raise ValueError(f'Shape not match! Can not add two matrix with shape {self.shape} and {B.shape}')
        ret_value = [self.element[i]-B.element[i] for i in range(0,len(self.element))]
        return Matrix(ret_value,self.shape)
    
    def __truediv__(self,B):
        '矩阵除以一个数'
        if not isinstance(B,(int,float,np.number,complex)):
            raise TypeError(f'Can not div a non-number!')
        if B == 0:
            raise ValueError(f'Can not div 0!')
        return self.__mul__(1/B)
    
    def __matmul__(self,B):
        '矩阵乘法 @'
        if self.shape[1]!=B.shape[0]:
            raise ValueError(f'Shape not match! Can not multiply two matrix with shape {self.shape} and {B.shape}')
        n,m,r = self.shape[0],self.shape[1],B.shape[1]
        ret_elm = [0]*n*r
        for i in range(0,n):
            for j in range(0,r):
                for k in range(0,m):
                    ret_elm[i*r+j] += self.element[i*m+k] * B.element[k*r+j]

        return Matrix(ret_elm,[n,r])
    
    def __mul__(self,B):
        '按位乘法 *'
        ret_elm = self.element
        if isinstance(B,Matrix):
            if self.shape[1]!=B.shape[0]:
                raise ValueError(f'Shape not match! Can not multiply two matrix with shape {self.shape} and {B.shape}')
            for i in range(0,self.shape[0]):
                for j in range(0,self.shape[1]):
                    ret_elm[i *self.shape[1] + j] = self.element[i *self.shape[1] + j] * B.element[i *self.shape[1] + j]
        if isinstance(B,(int,float,np.number,complex)):
            ret_elm = [ret_elm[i]*B for i in range(0,len(ret_elm))]
        
        return Matrix(ret_elm,self.shape)


    def reshape(self,n:int,m:int):
        '改变形状'
        if n*m!=len(self.element):
            raise ValueError('Shape invalid!')
        self.shape = [n,m]
    def trace(self):
        '迹'
        return sum([self.element[i*self.shape[1]+i] for i in range(0,self.shape[0])])

    def svd(self):
        '奇异值分解，此处不好实现，直接调库了'
        U,sigma,V = np.linalg.svd(np.mat(self.element).reshape(self.shape))
        return U,sigma,V

    def fro(self):
        'forbenius 范数，所有元素平方和开根号'
        return np.sqrt(sum([i**2 for i in self.element]))

    def L2(self):
        'L2 范数, 即最大的奇异值'
        _,sigma,__ = self.svd()
        return max(sigma)
    
    def L1(self):
        'L1，即最大列的绝对值和'
        n, m = self.shape[0], self.shape[1]
        elm_backup = [abs(i) for i in self.element]
        s = [sum(elm_backup[j::m]) for j in range(0,m)]
        return max(s)
    
    def Linf(self):
        'L无穷范数，即最大行绝对值和'
        n, m = self.shape[0], self.shape[1]
        elm_backup = [abs(i) for i in self.element]
        s = [sum(elm_backup[j*m:(j+1)*m]) for j in range(0,n)]
        return max(s)
    

class Eye(Matrix):
    def __init__(self, dim):
        elm = [0]*dim*dim
        for i in range(0,dim):
            elm[i*dim+i] = 1
        super().__init__(elm,[dim,dim])
def isClose(x,n=0):
    'x接近n?'
    return np.isclose(x,n)


def transpose(matrix):
    '转置'
    if not isinstance(matrix,Matrix):
        raise TypeError('Not a Matrix!')
    n,m = matrix.shape[0],matrix.shape[1]
    elm = [0]*n*m
    elm_conj = [i.conjugate() if isinstance(i,complex) else i for i in matrix.element ] # 共轭

    for i in range(0,m): # 转置
        elm[i*n:(i+1)*n] = elm_conj[i::m]
    return Matrix(elm,[m,n])

def minor(m,row,col):
    '返回余子式，去掉row行,col列'
    return Matrix([m.element[i] for i in range(0,len(m.element)) if int(i%m.shape[1])!=col and int(i/m.shape[1])!=row],[m.shape[0]-1,m.shape[1]-1])

def det(m):
    '计算行列式'
    if not isinstance(m,Matrix):
        raise TypeError('Only Matrix type object can calculate determinant!')
    if m.shape[0] != m.shape[1]:
        return 0
    if m.shape[0] == 2:
        '''
        |m[0] m[1]|
        |m[2] m[3]|
        '''
        return m.element[0]*m.element[3] - m.element[1]*m.element[2]
    ret = 0
    for i in range(0,m.shape[0]):
        '计算代数余子式'
        cof = m.element[i] * det(minor(m,0,i))
        ret += ((-1)**i) * cof
    return ret

def inv(matrix):
    '求逆'
    n,m = matrix.shape[0],matrix.shape[1]
    d = det(matrix)
    A_inv = [0]*n*n
    if n == m and d!=0:
        '方阵, 尝试用伴随矩阵求逆'
        A_elm = [0]*n*n
        for i in range(0,n):
            for j in range(0,n):
                sign = (-1)**(i+j)
                a = det(minor(matrix,i,j))
                A_elm[i*n+j] = sign*a
        A = Matrix(A_elm,[n,n])
        A_inv = transpose(A)/d # 矩阵的逆=伴随矩阵的转置/行列式
    else:
        raise NotImplementedError('没有逆!')
    return A_inv
        

def lu(matrix):
    'gauss 消元法进行LU分解'
    if not isinstance(matrix,Matrix):
        raise TypeError('Not a Matrix!')
    n,m = matrix.shape[0],matrix.shape[1]
    l_elm = [0]*n*m
    u_elm = copy.deepcopy(matrix.element)
    for i in range(0,n):
        t = u_elm[i*m:(i+1)*m] # 取出第i行的元素
        for j in range(i+1,n): # 用第i行元素消去i+1，i+2，...n-1行
            factor = u_elm[j*m+i] / u_elm[i*m+i] # 主元的比值u[j,i]/u[i,i]
            l_elm[j*m+i] = factor
            for k in range(0,m):
                u_elm[j*m+k] -= t[k]*factor
    for i in range(0,n):
        l_elm[i*m+i] = 1
    return Matrix(l_elm,matrix.shape),Matrix(u_elm,matrix.shape)


def qr(matrix,method='schmidt'):
    '''
    QR分解
    '''
    if not isinstance(matrix,Matrix):
        raise TypeError('Not a Matrix!')
    n,m = matrix.shape[0],matrix.shape[1]
    if n<m:
        raise NotImplementedError(f'{n}<{m}!')
    q_elm = copy.deepcopy(matrix.element)
    r_elm = [0]*n*m
    if method == 'schmidt':
        for i in range(0,m):
            q_i = Matrix([q_elm[k*m+i] for k in range(0,n)])
            backup = copy.deepcopy(q_i) # 备份
            for j in range(0,i):
                '''
                计算r(j,i) = q(j).T * q(i)
                q(j) = [q[k*m+j] for k in range(0,n)]
                q(i) = [q[k*m +i]]
                '''
                q_j = Matrix([q_elm[k*m+j] for k in range(0,n)])
                r = transpose(q_j) @ backup
                r = r.element[0]
                r_elm[j*m + i] = r 
                q_i = q_i - q_j*r # 减去已经正交化的分量的投影
            r = np.sqrt((transpose(q_i)@q_i).element[0]) # r = |q_i|
            r_elm[i*m+i] = r # r(i,i)
            q_i = q_i/r # 归一化 q_i = q_i / |q_i|
            q_elm[i::m] = q_i.element # 赋值回要返回的变量

        q_elm_backup = copy.deepcopy(q_elm)
        q_elm = [0]*n*n
        for i in range(0,n): # 此时的q_elm 可能只有n*m，但是我们要将其补齐到n*n
            for j in range(0,m):
                q_elm[i*n:(i+1)*n] = q_elm_backup[i*m:(i+1)*m] + [0]*(n-m)
        r_elm = r_elm + [0]*(n-m)*m # 同理，r_elm可能只有m*m，需要将其补零到n*m


                
    elif method == 'householder':
        backup = copy.deepcopy(q_elm)
        R = copy.deepcopy(matrix)
        P = Eye(n)
        t = 0
        dim = min(n,m)
        for i in range(0,dim):
            u = Matrix(backup[0::m-t]) # u <- x_1
            r = np.sqrt((transpose(u)@u).element[0]) # r <- |u|
            u = u - Matrix(Eye(n-t).element[0::n-t])*r # u <- u - e_1*|u|
            if isClose(sum(u.element)):
                P_t_ = Eye(n-t)
            else:
                P_t_ = Eye(n-t) - u@transpose(u)/(transpose(u)@u).element[0]*2 # R = I - 2*(u@u.T)/(u.T@u)
            P_t = Eye(n)
            for j in range(0,n-t):
                for k in range(0,n-t):
                    P_t.element[(j+t)*n+k+t] = P_t_.element[j*(n-t) + k] # 还原完整的P_t矩阵，将P_t_左上角padding即可
            P = P_t@P # PA = R, Q = inv(P) = P.T
            R = P_t @ R
            t += 1
            if dim-t !=0:
                backup = copy.deepcopy(R)
                for i in range(0,t):
                    backup = minor(backup,0,0)
                backup = backup.element
        r_elm = R.element
        q_elm = transpose(P).element
        
    elif method == 'givens':
        backup = copy.deepcopy(q_elm)
        R = copy.deepcopy(matrix)
        P = Eye(n)
        t = 0
        for i in range(0,m):
            u = Matrix(backup[0::m-t]) # u <- x_1
            P_t_ = Eye(n-t) # 临时存放P矩阵，且是切片
            P_t = Eye(n) # 临时存放完整的P矩阵，是P_t_ padding左上角之后的矩阵
            for j in range(1,n-t): # 依次消去u[0]后面的所有元素
                r = np.sqrt(u.element[0]**2 + u.element[j]**2)
                c = u.element[0]/r
                s = u.element[j]/r
                P_t_ = Eye(n-t)
                P_t_.element[0*(n-t)+0] = c
                P_t_.element[0*(n-t)+j] = s
                P_t_.element[j*(n-t)+j] = c
                P_t_.element[j*(n-t)+0] = -s

                for j in range(0,n-t): # 此时的P_t_就是将第i列三角化的旋转矩阵的切片其维度为[dim,dim]，我们将其还原为P_t
                    for k in range(0,n-t):
                        P_t.element[(j+t)*n+k+t] = P_t_.element[j*(n-t) + k] # 还原完整的Q矩阵，将Q_T左上角padding即可
                R = P_t @ R # 此时的P_t可以将R矩阵的前i列三角化
                u = Matrix(R.element[0::m-t])
                P = P_t @ P # 更新P矩阵
            t += 1
            if m-t !=0:# 此时R的前i列已经完成三角化，其余子式R*[0,0]恰好是下一个要三角化的切片
                backup = copy.deepcopy(R)
                for i in range(0,t):
                    backup = minor(backup,0,0)
                backup = backup.element 
                
        r_elm = R.element
        q_elm = transpose(P).element
    elif method == 'modified schmidt':
        '''
        相比于normal schmidt 约简，每一步都更新后面所有的q_j，但每次只归一化q_i
        '''
        u = [0]*m
        E = [0]*m
        u[0] = Matrix(q_elm[0::m]) # 取出第0列
        u[0] = u[0]/np.sqrt((transpose(u[0])@u[0]).element[0]) # u_1 <- x_1/|x_1|
        for j in range(1,m):
            u[j] = Matrix(q_elm[j::m]) # u_j <- x_j
        for k in range(1,m):
            E[k] = Eye(m) - u[k-1]@transpose(u[k-1])
            for j in range(k,m):
                u[j] = E[k]@u[j]
            u[k] = u[k]/np.sqrt((transpose(u[k])@u[k]).element[0])

        for i in range(0,m):
            for j in range(0,n):
                q_elm[j*m+i] = u[i].element[j]
        
        q_elm_backup = copy.deepcopy(q_elm)
        q_elm = [0]*n*n
        for i in range(0,n): # 此时的q_elm 可能只有n*m，但是我们要将其补齐到n*n
            for j in range(0,m):
                q_elm[i*n:(i+1)*n] = q_elm_backup[i*m:(i+1)*m] + [0]*(n-m)
        r_elm = (transpose(Matrix(q_elm,[n,n]))@matrix).element

    else:
        raise NotImplementedError('Unknown method! Supported methods: [schmidt, householder, givens, modified schmidt].')
    Q = Matrix(q_elm,[n,n])
    R = Matrix(r_elm,[n,m])
    return Q,R
            
        
def urv(matrix):
    '''
    URV分解，设待分解矩阵为A
    首先用householder分解P@A = (B,0).T
    再对B.T进行householder分解，使得Q@(B.T) = (C,0).T
    此时我们有，A = P.T@(B,0).T = P.T@(C.T,0;0,0)@Q
    即有URV分解:
    U = P, R = (C.T,0;0,0), V.T = Q
    '''
    if not isinstance(matrix,Matrix):
        raise TypeError('Not a Matrix!')
    n,m = matrix.shape[0],matrix.shape[1]
    Q_1,R_1 = qr(matrix,'householder') # qr分解A, A = Q_1 @ R_1, Q_1维度为(n,n),R_1维度为(n,m)
    U = Q_1 # U = Q_1, U维度为(n,n)
    b_elm = [] # 临时存放B的element
    r = 0
    for i in range(0,n): # R_1的维度为(n,m),我们要找出其中的不为零的部分即形状为(r,m)的B矩阵
        t = R_1.element[i*m:(i+1)*m] # 取出第i行的元素，看其是否全为0
        if not isClose(sum(t)):
            b_elm += t # 将不全为零行添加到临时变量中
        else:
            break
        r += 1 # r为当前不为零的行数
    B = Matrix(b_elm,[r,m])
    Q_2,R_2 = qr(transpose(B),'householder') # Q_2 @ R_2 = B.T, 其中B.T维度为(m,r), Q_2维度为(m,m), R_2维度为(m,r)
    V = transpose(Q_2) # V = Q_2.T, V维度为(m,m)
    c_elm = [] # 临时存放C的element
    e = 0
    for i in range(0,m): # R_2维度为(m,r)，我们要找出其中上部分形状为(e,r)
        e += 1
        t = R_2.element[i*r:(i+1)*r] # 取出第i行的元素，看其是否为0
        if not isClose(sum(t)):
            c_elm += t
        else:
            break
    
    C = transpose(Matrix(c_elm,[r,r])) # 其实e=r
    c_elm = C.element
    r_elm = []
    for i in range(0,n): # 此时的C维度为(r,r),所以还需要padding补零, 补为(n,m)
        if i < r:
            r_elm += c_elm[i*r:(i+1)*r] + [0]*(m-r) # 每一行都补m-r个零
        else:
            r_elm += [0]*m
    
    R = Matrix(r_elm,[n,m])
    return U,R,V




def solve(A,b,method='lu',leastSquare=True):
    '''
    求解Ax=b,
    leastSquare = True 表示，可以求最小二乘解
    '''
    d = det(A)
    if d == 0: # 行列式为零，求最小二乘解
        if leastSquare:
            print('det(A)=0,求最小二乘解.')
            b = transpose(A)@b
            A = transpose(A)@A
            d = det(A)
        else:
            raise ValueError('|A|=0!')
    n = A.shape[0]
    m = A.shape[1]
    if not (isinstance(A,Matrix) or isinstance(b,Matrix)):
        raise TypeError('Not a Matrix')
    if n != b.shape[0]:
        raise ValueError(f'Shape not match! A:{A.shape},b:{b.shape}')
    x = [0]*n

    if method == 'lu':
        '''
        A = LU
        LUx=b
        Ly = b
        Ux = y
        '''
        L,U = lu(A)
        l_elm = L.element
        u_elm = U.element
        y = copy.copy(b.element)
        for i in range(0,n):
            for j in range(0,i):
                y[i] -= y[j]*l_elm[i*m+j]
            y[i] /= l_elm[i*m+i]
        x = y
        for i in range(n-1,-1,-1):
            for j in range(n-1,i,-1):
                x[i] -= x[j]*u_elm[i*m+j]
            x[i] /= u_elm[i*m+i]

    elif method == 'qr':
        '''
        A=QR
        QRx=b
        Rx=Q*B
        '''
        Q,R = qr(A,'householder')
        y = transpose(Q)@b # y = Q*b
        x = y.element
        r_elm = R.element
        for i in range(n-1,-1,-1):
            for j in range(n-1,i,-1):
                x[i] -= x[j]*r_elm[i*m+j]
            x[i] /= r_elm[i*m+i]
    elif method == 'cramer':
        '''
        x_i = det(A_i)/det(A)
        '''
        x = [0]*n
        for i in range(0,n):
            A_i= copy.deepcopy(A)
            A_i.element[i::m] = b.element
            x[i] = det(A_i)/d

    return Matrix(x)

