import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AppGlobalConstants } from 'src/app/app.constants';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DashboardService {

  constructor(private http: HttpClient,private globalConst: AppGlobalConstants) {
  
   }

   
 
   postImageData(skudata):Observable<any> {
     return this.http.get(this.globalConst.W_API_URL + 'skuanalysis_filterchange?' + skudata );

    //local
    //  return this.http.get(this.globalConst.WEB_API_URL + 'skuanalysis_filterchange?' + skudata );
   }

 
}
