import fs from 'fs';

fs.readdir('../raw', (err, files) => {
    if(err){
        console.log(err);
    }else{
        for(let file of files){
            
        }
    }
});
