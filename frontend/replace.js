const fs = require('fs');
const path = require('path');

function walk(dir) {
    let results = [];
    const list = fs.readdirSync(dir);
    list.forEach(function (file) {
        file = path.join(dir, file);
        const stat = fs.statSync(file);
        if (stat && stat.isDirectory()) {
            results = results.concat(walk(file));
        } else {
            if (file.endsWith('.ts')) results.push(file);
        }
    });
    return results;
}

const files = walk('d:/fyp/sem1_finalized_malaika/sem1/frontend/app/api');
let count = 0;

files.forEach(file => {
    let content = fs.readFileSync(file, 'utf8');
    let changed = false;

    if (content.match(/process\.env\.FLASK_API_URL\s*\|\|\s*'http:\/\/localhost:5000'/)) {
        content = content.replace(/process\.env\.FLASK_API_URL\s*\|\|\s*'http:\/\/localhost:5000'/g, "process.env.FLASK_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'");
        changed = true;
    }

    if (content.includes('`http://localhost:5000/api')) {
        content = content.replace(/`http:\/\/localhost:5000\/api/g, "`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'}/api");
        changed = true;
    }

    if (changed) {
        fs.writeFileSync(file, content, 'utf8');
        count++;
    }
});

console.log('Updated ' + count + ' files');
