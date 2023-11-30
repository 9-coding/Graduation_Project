import React, { useState, useEffect } from 'react';
import { View, Image, Button, StyleSheet, Text, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
//import firebase from 'firebase/app';
//import 'firebase/storage';
import { storage } from "../firebaseConfig";
import { ref, uploadBytes, getDownloadURL } from 'firebase/compat/storage';




export default function ImageUploadScreen() {
    const [selectedImage, setSelectedImage] = useState(null);

    //권한 허용
    useEffect(() => {
        (async () => {
            if (Platform.OS !== 'web') {
                const cameraStatus = await ImagePicker.requestCameraPermissionsAsync();
                const mediaLibraryStatus = await ImagePicker.requestMediaLibraryPermissionsAsync();

                if (cameraStatus.status !== 'granted' || mediaLibraryStatus.status !== 'granted') {
                    alert('Permission to access camera and media library is required!');
                }
            }
        })();
    }, []);

    //갤러리에서 사진 선택
    const pickImage = async () => {
        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });

        if (!result.canceled) {
            setSelectedImage(result.assets[0].uri);
        }

    };

    //카메라로 촬영
    const takePicture = async () => {
        let result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });

        if (!result.canceled) {
            setSelectedImage(result.assets[0].uri);
        }
    };

    //이미지 업로드
    const uploadImage = async () => {
        try {
            const response = await fetch(selectedImage);
            const blob = await response.blob();

            // Create a reference to the storage bucket
            const storageRef = firebase.storage().ref();

            // Create a unique filename for the image
            const filename = `${Date.now()}.jpg`;

            // Upload the image to Firebase Storage
            const uploadTask = storageRef.child(`Cloth/${filename}`).put(blob);

            uploadTask.on(
                'state_changed',
                (snapshot) => {
                    // Track upload progress if needed
                    const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
                    console.log(`Upload is ${progress}% done`);
                },
                (error) => {
                    console.error('Error uploading image:', error);
                },
                () => {
                    // Handle successful upload
                    console.log('Image uploaded successfully!');
                }
            );
        } catch (error) {
            console.error('Error preparing image for upload:', error);
        }
    };

   

    return (
        <View style={styles.container}>
            <Button title="Take Picture" onPress={takePicture} />
            <Button title="Pick Image from Gallery" onPress={pickImage} />
            <Button title="Upload Image" onPress={uploadImage} />
            {selectedImage && <Image source={{ uri: selectedImage }} style={styles.image} />}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    image: {
        width: 200,
        height: 200,
        marginTop: 20,
    },
});
