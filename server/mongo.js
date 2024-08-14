const mongoose = require('mongoose');

const SignupSchema = new mongoose.Schema({
    fname: { type: String, required: true },
    lname: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true }
});

const SignupCollection = mongoose.model('SignupCollection', SignupSchema);

module.exports = SignupCollection;
