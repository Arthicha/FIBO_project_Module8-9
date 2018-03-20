


f = open("C:\\Users\cha45\PycharmProjects\FIBO_project_Module8-9\Dataset\Tew\project\histogram_dataset_0_test.txt",'r')
data = f.read()
f.close()
data=data.split('\n')
data=data[:-1]
data = list(map(lambda x:x.split(','),data))
data = list(map(lambda x:list(map(lambda y: float(y),x)),data))
print()
