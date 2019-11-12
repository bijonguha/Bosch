# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:22:29 2019
@author: HRO6KOR
"""

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from app import *
import cv2
import os
from requests_toolbelt import MultipartEncoder
from flask import make_response
import resources.algo_final as algo
from flask import jsonify

UPLOAD_FOLDER = 'in/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)
upload_parser.add_argument('A', location='args', required=True)
upload_parser.add_argument('B', location='args', required=True)
upload_parser.add_argument('X', location='args', required=True)
upload_parser.add_argument('Y', location='args', required=True)

@ns_conf.route('/upload')
@ns_conf.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args() 
        A = float(args['A'])
        B = float(args['B'])
        X = float(args['X'])
        Y = float(args['Y'])
        uploaded_file = args['file'] 
        image = uploaded_file.filename
        content_type = uploaded_file.headers["Content-Type"]
        filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imagepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if content_type == "image/jpeg":
            print("TRUE") 
            nameofimages = algo.checker(imagepath,A,B,X,Y)
            if type(nameofimages) == list:
                print("LIST" )
                os.remove("in/"+image)
                response ={"image1" :  str(nameofimages[0])  , "image2" : str(nameofimages[1]) , "image3" :str(nameofimages[2]) }
                print("RESPONSE" , response )
                resp = make_response(jsonify(response))
                print("RESULT",resp)
                resp.headers.extend({"Access-Control-Allow-Origin": "*"}) 
                return resp   
            else:
                resp = make_response(jsonify(nameofimages),404)
                resp.headers.extend({"Access-Control-Allow-Origin": "*"})
                return resp
    


delete_parser = api.parser()
delete_parser.add_argument('image1',location='args',required=True)
delete_parser.add_argument('image2',location='args',required=True)
delete_parser.add_argument('image3',location='args',required=True)
@ns_conf.route('/delete')
@ns_conf.expect(delete_parser)
class DeleteImages(Resource):
    def get(self):
        args = delete_parser.parse_args()
        image1 = args['image1']
        image2 = args['image2']
        image3 = args['image3']
        os.remove("static/"+image1)
        os.remove("static/"+image2)
        os.remove("static/"+image3)
        resp = make_response({"result" : "Deleted all images successfully"})
        resp.headers.extend({"Access-Control-Allow-Origin": "*"})
        return resp
