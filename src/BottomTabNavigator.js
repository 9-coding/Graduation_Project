import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import HomeScreen from './HomeScreen'; // Home 화면 컴포넌트를 import
import CalendarScreen from './CalendarScreen'; // Calendar 화면 컴포넌트를 import
import AddScreen from './AddScreen'; // Add 화면 컴포넌트를 import
import ClosetScreen from './ClosetScreen'; // Closet 화면 컴포넌트를 import
import ProfileScreen from './ProfileScreen'; // Profile 화면 컴포넌트를 import

const Tab = createBottomTabNavigator();

const BottomTabNavigator = () => {
    return (
        <Tab.Navigator
            screenOptions={({ route }) => ({
                tabBarIcon: ({ focused, color, size }) => {
                    let iconName;

                    if (route.name === 'Home') {
                        iconName = focused ? 'home' : 'home-outline';
                    } else if (route.name === 'Calendar') {
                        iconName = focused ? 'calendar' : 'calendar-outline';
                    } else if (route.name === 'Add') {
                        iconName = focused ? 'add-circle' : 'add-circle-outline';
                    } else if (route.name === 'Closet') {
                        iconName = focused ? 'shirt' : 'shirt-outline';
                    } else if (route.name === 'Profile') {
                        iconName = focused ? 'person' : 'person-outline';
                    }

                    // 아이콘 이름에 따라 Ionicons 아이콘을 표시
                    return <Ionicons name={iconName} size={size} color={color} />;
                },
            })}
            tabOptions={{
                activeTintColor: 'black', // 선택된 탭의 색상
                inactiveTintColor: 'gray', // 선택되지 않은 탭의 색상
            }}
        >
            <Tab.Screen name="Home" component={HomeScreen} />
            <Tab.Screen name="Calendar" component={CalendarScreen} />
            <Tab.Screen name="Add" component={AddScreen} />
            <Tab.Screen name="Closet" component={ClosetScreen} />
            <Tab.Screen name="Profile" component={ProfileScreen} />
        </Tab.Navigator>
    );
};

export default BottomTabNavigator;