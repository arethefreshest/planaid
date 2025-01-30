import React from 'react';
import { View, StyleSheet } from 'react-native';
import MapView from 'react-native-maps';

const CustomMap: React.FC = () => {
  return (
    <View style={styles.container}>
      <MapView style={styles.map} />
    </View>
  );
};

export default CustomMap;

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  map: {
    width: '100%',
    height: '100%'
  }
});
