from enum import Enum

class Response(Enum):
    
    FILE_TYPE_NOT_SUPPORTED = "File_Type_Not_Supported"
    FILE_UPLOADED_SUCCESS = "File_Uploaded_Success"
    FILE_UPLOADED_FAILED = "File_Uploaded_Failed"
    PROCESSING_SUCCESS= "Processing Success"
    PROCESSING_FAILED= "Processing Failed"
    CHAT_SUCCESS= "Chat Success"
