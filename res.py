def apiresponse(status,data=None,message=""):
   return {"detail":{"data":{"status code":status,"prediction":data,"message":message},"errors":None}}



def apierror(status,message=""):
   return {"detail":{"errors":{"status code":status,"message": message},"data":None}}