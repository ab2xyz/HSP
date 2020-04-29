import traceback


class SomeObject():


 def __init__(self, def_name=None):


     if def_name == None:


         (filename,line_number,function_name,text)=traceback.extract_stack()[-2]


         def_name = text[:text.find('=')].strip()


         self.defined_name = def_name



ThisObject = SomeObject()


print(ThisObject.defined_name)


ThatObject= SomeObject()

print(ThatObject.defined_name)

hisObject=ThatObject
print(hisObject.defined_name)
