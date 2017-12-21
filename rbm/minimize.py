import numpy as np
import fdf
def get_minimize(f, df, data, X, Dim, max_iter):
	INT = 0.1; EXT = 3.0; MAX = 20; RATIO = 10; SIG = 0.1; RHO = SIG/2
	red = 1; length = max_iter; S = 'Linesearch'
	i = 0; ls_failed = 0; f0 = f; df0 = df; fX = f0
	i = i + (length<0)
	#s(n,1), d0, x3 is a number
	s = -df0; d0=-np.dot(s.T,s); x3=red/(1-d0)

	while i < abs(length):
		i = i + (length>0)
		#X(n,1), f0,F0 is a number, dF0, df0(n,1)
		X0 = X;F0 = f0;dF0 = df0

		if length > 0:
			M = MAX
		else:
			M = min(MAX, -length-i)
		while 1:
			#x2, f2, d2, f3 is a number,  df3(n,1)
			x2 = 0;f2 = f0;d2 = d0;f3 = f0;df3 = df0
			success = 0
			while (not success) and M > 0:
				try:
					M = M-1;i=i+(length<0)
					f3,df3 = fdf.get_fdf(data, X+x3*s, Dim)
				except:
					x3 = (x2+x3)/2

			if f3<F0:
				X0 = X+x3*s
				F0 = f3
				dF0 = df3
			#d3 is a number
			d3 = np.dot(df3.T, s)
			if d3>SIG*d0 or f3>f0+x3*RHO*d0 or M==0:
				break
			x1=x2; f1=f2; d1=d2;
			x2=x3; f2=f3; d2=d3;
			A=6*(f1-f2)+3*(d2+d1)*(x2-x1)
			B=3*(f2-f1)-(2*d1+d2)*(x2-x1)
			x3 = x1-d1*(x2-x1)**2/(B+(B*B-A*d1*(x2-x1))**0.5)
			if not (np.isreal(x3)) or np.isnan(x3) or np.isinf(x3) or x3<0:
				x3 = x2*EXT
			elif x3 > x2*EXT:
				x3 = x2*EXT
			elif x3 < x2+INT*(x2-x1):
				x3 = x2+INT*(x2-x1)

		while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:
			if d3>0 or f3>f0+x3*RHO*d0:
				x4=x3;f4=f3;d4=d3;
			else:
				x2=x3;f2=f3;d2=d3;

			if f4>f0:
				x3=x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
			else:
				A=6*(f2-f4)/(x4-x2)+3*(d4+d2)
				B=3*(f4-f2)-(2*d2+d4)*(x4-x2)
				x3 = x2 + ((B*B-A*d2*(x4-x2)**2)**0.5-B)/A

			if np.isnan(x3) or np.isinf(x3):
				x3 = (x2+x4)/2

			x3 = max(min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2))
			f3,df3 = fdf.get_fdf(data, X+x3*s, Dim)
			if f3<F0:
				X0 = X+x3*s
				F0=f3
				dF0 = df3

			M = M-1
			i=i+(length<0)
			d3=np.dot(df3.T,s)

		if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:
			X = X+x3*s; f0=f3; 
			#fX=np.concatenate((fX.T, f0), axis=1).T
			print('{} {} ; Value:{} \r'.format(S,i,f0))
			s = (np.dot(df3.T,df3)-np.dot(df0.T,df3))/np.dot(df0.T,df0)*s - df3
			df0 = df3
			d3=d0; d0=np.dot(df0.T,s)
			if d0 > 0:
				s=-dF0
				d0 = -np.dot(s.T,s)
			x3 = x3*min(RATIO,d3/(d0-realmin))
			ls_failed=0
		else:
			X=X0;f0=F0;df0=dF0;
			if ls_failed or i>abs(length):
				break;

			s = -df0;d0=-np.dot(s.T,s);
			x3 = 1/(1-d0)
			ls_failed = 1
	print('\n')

	return X
