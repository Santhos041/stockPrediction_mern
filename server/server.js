const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const SignupCollection = require('./mongo');

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

mongoose.connect("mongodb://localhost:27017/stockprediction_mern")
    .then(() => console.log("MongoDB connected"))
    .catch(err => console.log("Failed to connect to MongoDB:", err));

app.post('/Signup', async (req, res) => {
    const { email, password, fname, lname } = req.body;
    const data = {
        email: email,
        password: password,
        fname: fname,
        lname: lname
    };

    try {
        const check = await SignupCollection.findOne({ email: email });
        if (check) {
            res.json("User already exists");
        } else {
            await SignupCollection.create(data);
            res.json("Signup success");
        }
    } catch (e) {
        console.error(e);
        res.json("Error occurred");
    }
});

app.listen(8000, () => {
    console.log('Server is running on port 8000');
});
app.post('/Login',async(req,res)=>{
    const {email,password}=req.body;
    try {
        const user = await SignupCollection.findOne({ email: email });
        
        if (user) {
            if (user.password === password) {
                res.json("Login success");
            } else {
                res.json("Invalid password");
            }
        } else {
            res.json("User not found");
        }
    } 
    
    catch(e){
        res.json("notexist")
    }
})
