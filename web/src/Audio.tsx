import H5AudioPlayer from "react-h5-audio-player";
import 'react-h5-audio-player/lib/styles.css';


export function Audio ({ className = '', ...rest }: typeof H5AudioPlayer.defaultProps) {
  return (
    <H5AudioPlayer
      className={`rounded-b-lg rounded-tr-lg bg-gray-50 ${className}`}
      style={{ boxShadow: 'none', maxWidth: 400 }}
      autoPlay={true}
      {...rest}
    />
  )
}
