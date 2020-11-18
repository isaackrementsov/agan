import fs from 'fs';
import uuid4 from 'uuid';
import request from 'request';
import path from 'path';

const __dirname = path.resolve();
const start = new Date().getTime();
const msg = 'Valid data not found in urls.json! Run webScraper.js first.';

const download = (url, path) => {
    request.head(url, (_err, _res, _body) => {
        request(url).pipe(fs.createWriteStream(path))
    });
}

fs.readFile('urls.json', (err, data) => {
    if(err){
        console.log(msg);
    }else{
        if(data){
            try {
                // Get URLs from JSON file
                data = JSON.parse(data);
                const urls = data.urls;

                for(let url of urls){
                    // Download data from each url and save to a file in ../raw directory
                    const parts = url.split('/');
                    const l = parts.length - 1;
                    const filename = parts[l - 1] + parts[l];

                    download(url, path.join(__dirname, '../raw/' + filename));
                }

            }catch(e){
                console.log(e);
                console.log(msg);
            }
        }
    }
});
