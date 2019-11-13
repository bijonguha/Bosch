import { Component, OnInit} from '@angular/core';
import { DashboardService } from './dashboard.service';
import { HttpClient } from '@angular/common/http';
@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {


  SERVER_URL = 'http://13.233.58.89:5005/api/OCR/upload?';
  aValue; bValue; xValue; yValue;
  file;
  fileName;
  jsondata; jsondata1; jsondata2;
  loaderhide_Show = false; 
responseErrortext;
noDataFound =  false;


  constructor(private httpService: HttpClient) {}

  ngOnInit() { }

  onFileSelect(event) {
     
    if (event.target.files.length > 0) {
      const file = event.target.files[0];
      
      this.fileName = file;
      this.file = event.target.files[0].name;
    
    }
  }
  onSubmit() {
    this.loaderhide_Show = true;
    const formData = new FormData();
    formData.append('file', this.fileName);
    //alert(JSON.stringify(this.filename));
    this.httpService.post<any>(this.SERVER_URL + 'A=' + this.aValue + '&B=' + this.bValue + "&X=" + this.xValue + "&Y=" + this.yValue, formData).subscribe(
      response => {
 this.noDataFound  =  false;
       
        this.jsondata = 'http://13.233.58.89:5005/static/' +response.image1;
        this.jsondata1 = 'http://13.233.58.89:5005/static/' +response.image2;
        this.jsondata2 = 'http://13.233.58.89:5005/static/' +response.image3;
        this.loaderhide_Show = false;       

      },
      (err) => {
       console.log(JSON.stringify(err));             
if(err.status == '404'){
           this.noDataFound  =  true;  
           this.responseErrortext = err.error.response;
}
if(err.status == '0'){
           this.noDataFound  =  true;  
    this.responseErrortext = "Http failure response - 403 Forbidden";
}
 console.log(""+err.status);
        this.loaderhide_Show = false;   
      }
    );
  }
}
